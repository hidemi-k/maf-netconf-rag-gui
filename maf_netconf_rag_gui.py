# Copyright (c) 2026 hidemi-k / Licensed under the MIT License
# maf-netconf-rag-gui: NiceGUI frontend for NETCONF × Agentic RAG (MAF Phase 7)
#
# Architecture: Orchestrator-Worker pattern
#   Orchestrator → task_decomposer → dependency_resolver → Worker(s) → result_aggregator
#   Worker: get_inventory → translate → Generator(ReAct) → Reviewer → validate → fix → deploy → audit → rollback

import asyncio
import os
import re
import copy
import json
import yaml
import configparser
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field as dc_field

# LangChain / FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Junos
try:
    from jnpr.junos import Device
    from jnpr.junos.utils.config import Config
    from jnpr.junos.exception import RpcError, ConnectError, ConfigLoadError, CommitError
    JUNOS_AVAILABLE = True
except ImportError:
    JUNOS_AVAILABLE = False

# Agent Framework (MAF)
from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatClient

from nicegui import ui

# ─────────────────────────────────────────────────────────────────
# 1. Configuration & LLM Setup
# ─────────────────────────────────────────────────────────────────

config = configparser.ConfigParser()
config_paths = [
    './config.ini',
    os.path.expanduser('~/config/config.ini'),
]

GROQ_API_KEY = None
for path in config_paths:
    if os.path.exists(path):
        config.read(path)
        if 'GROQ' in config and 'GROQ_API_KEY' in config['GROQ']:
            GROQ_API_KEY = config['GROQ']['GROQ_API_KEY']
            break

if not GROQ_API_KEY:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

GROQ_BASE_URL  = "https://api.groq.com/openai/v1"
DEFAULT_MODEL  = "llama-3.3-70b-versatile"

# Load embeddings & vector store
embedding  = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_db")
if os.path.exists(FAISS_PATH):
    vectorstore = FAISS.load_local(FAISS_PATH, embedding, allow_dangerous_deserialization=True)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    retriever = None

llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)

def make_client(model_id: str = DEFAULT_MODEL) -> OpenAIChatClient:
    return OpenAIChatClient(model=model_id, api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# ─────────────────────────────────────────────────────────────────
# 2. Skill Definitions
# ─────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = dc_field(default_factory=dict)

    def execute(self, **kwargs) -> Any:
        return self.function(**kwargs)


# ── Skill 1: validate_xml ────────────────────────────────────────
def validate_xml_structure(xml_config: str) -> Dict[str, Any]:
    if not xml_config or not xml_config.strip():
        return {"valid": False, "errors": ["XML is empty"], "warnings": []}
    try:
        root = ET.fromstring(xml_config)
        errors, warnings = [], []
        if root.tag != 'configuration':
            errors.append(f"Root must be <configuration>, got <{root.tag}>")
        for elem in root.iter():
            if elem.tag.isdigit():
                errors.append(f"Numeric-only tag name forbidden: <{elem.tag}>")
        if root.find('vlan') is not None:
            errors.append("<vlan> must be under <vlans>: missing <vlans> parent tag")
        for vlan in root.findall('.//vlan'):
            op_tag = vlan.find('operation')
            if op_tag is not None:
                errors.append('<operation> must be an XML attribute, not a child tag.')
        for vlan in root.findall('.//vlan[@operation="delete"]'):
            if vlan.find('vlan-id') is not None:
                errors.append("Delete operation must NOT contain <vlan-id>")
            if vlan.find('name') is None and vlan.find('n') is None:
                errors.append("Delete operation must have <name> tag with VLAN name")
        for vlan in root.findall('.//vlan'):
            if vlan.get('operation') != 'delete' and vlan.find('operation') is None:
                if vlan.find('name') is None and vlan.find('n') is None:
                    warnings.append("Create operation: missing <name> tag")
                if vlan.find('vlan-id') is None:
                    warnings.append("Create operation: missing <vlan-id> tag")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    except ET.ParseError as e:
        return {"valid": False, "errors": [f"XML syntax error: {str(e)}"], "warnings": []}


# ── Skill 2: fix_xml ─────────────────────────────────────────────
def fix_xml_structure(xml_config: str, translated_query: str = "") -> Dict[str, Any]:
    if not xml_config:
        return {"fixed_xml": xml_config, "changes": ["No XML to fix"], "success": False}
    changes = []
    try:
        root = ET.fromstring(xml_config)
        vlan_direct = root.find('vlan')
        if vlan_direct is not None:
            vlans_elem = ET.Element('vlans')
            root.remove(vlan_direct)
            vlans_elem.append(vlan_direct)
            root.append(vlans_elem)
            changes.append("Fixed: Added missing <vlans> wrapper around <vlan>")
        for vlan in root.findall('.//vlan'):
            op_tag = vlan.find('operation')
            if op_tag is not None:
                op_value = (op_tag.text or '').strip()
                vlan.set('operation', op_value)
                vlan.remove(op_tag)
                changes.append(f"Fixed: Converted <operation>{op_value}</operation> to attribute")
        for vlan in root.findall('.//vlan[@operation="delete"]'):
            n_tag = vlan.find('n')
            if n_tag is not None:
                name_tag = ET.Element('name')
                name_tag.text = n_tag.text
                idx = list(vlan).index(n_tag)
                vlan.remove(n_tag)
                vlan.insert(idx, name_tag)
                changes.append(f"Fixed: Renamed <n> to <name>")
            name_elem = vlan.find('name')
            vlan_name = name_elem.text if name_elem is not None else None
            for child in list(vlan):
                if child.tag != 'name':
                    vlan.remove(child)
                    changes.append(f"Fixed: Removed <{child.tag}> from delete operation")
            if vlan.find('name') is None and vlan_name:
                name_tag = ET.SubElement(vlan, 'name')
                name_tag.text = vlan_name
        fixed_xml = ET.tostring(root, encoding='unicode')
        return {"fixed_xml": fixed_xml, "changes": changes if changes else ["No changes needed"], "success": True}
    except ET.ParseError as e:
        return {"fixed_xml": xml_config, "changes": [f"XML parse error: {str(e)}"], "success": False}


# ── Skill 3: deploy_netconf ──────────────────────────────────────
def deploy_netconf_config(
    xml_config: str, device_ip: str, username: str, password: str,
    port: str = "830", comment: str = "AI Agent - MAF GUI"
) -> Dict[str, Any]:
    if not JUNOS_AVAILABLE:
        return {"status": "skipped", "diff": "", "message": "Junos modules not available"}
    try:
        with Device(host=device_ip, user=username, password=password, port=int(port)) as dev:
            cu = Config(dev)
            try:
                cu.load(xml_config, format="xml", merge=True)
            except ConfigLoadError as load_err:
                severity = getattr(load_err, 'severity', 'error')
                err_msg  = getattr(load_err, 'message', str(load_err))
                is_not_found = "statement not found" in err_msg.lower()
                if severity == 'warning' or is_not_found:
                    cu.rollback()
                    return {"status": "no_changes", "diff": "", "message": f"Target not found: {err_msg}"}
                cu.rollback()
                return {"status": "failure", "diff": "", "message": f"ConfigLoadError: {err_msg}"}
            diff = cu.diff()
            if diff:
                cu.commit(comment=comment)
                return {"status": "success", "diff": diff, "message": f"Deployed to {device_ip}"}
            return {"status": "no_changes", "diff": "", "message": "No configuration changes detected"}
    except Exception as e:
        return {"status": "failure", "diff": "", "message": f"Deployment failed: {str(e)}"}


# ── Skill 4: rollback ────────────────────────────────────────────
def rollback_config(device_ip: str, username: str, password: str,
                    port: str = "830", mode: str = "candidate") -> Dict[str, Any]:
    if not JUNOS_AVAILABLE:
        return {"status": "skipped", "mode": mode, "message": "Junos modules not available"}
    try:
        with Device(host=device_ip, user=username, password=password, port=int(port)) as dev:
            cu = Config(dev)
            if mode == "candidate":
                cu.rollback(0)
                return {"status": "success", "mode": "candidate", "message": "Candidate configuration discarded"}
            elif mode == "rescue":
                cu.rescue(action="reload")
                cu.commit(comment="Rollback to rescue config by MAF Agent")
                return {"status": "success", "mode": "rescue", "message": "Restored to rescue configuration"}
            return {"status": "failure", "mode": mode, "message": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"status": "failure", "mode": mode, "message": f"Rollback failed: {str(e)}"}


# ── Skill 5: audit ───────────────────────────────────────────────
def audit_deployment(xml_config: str, device_ip: str, username: str,
                     password: str, port: str = "830") -> Dict[str, Any]:
    if not JUNOS_AVAILABLE:
        return {"status": "skipped", "operation": "unknown", "vlan_name": "", "evidence": "",
                "message": "Junos modules not available"}
    try:
        root = ET.fromstring(xml_config)
        vlan = root.find('.//vlan')
        if vlan is None:
            return {"status": "failure", "operation": "unknown", "vlan_name": "",
                    "evidence": "", "message": "No <vlan> element found"}
        res = vlan.find('name')
        name_tag = res if res is not None else vlan.find('n')
        vlan_name = name_tag.text.strip() if name_tag is not None else ""
        operation = "delete" if vlan.get('operation') == "delete" else "create"
    except ET.ParseError as e:
        return {"status": "failure", "operation": "unknown", "vlan_name": "", "evidence": "",
                "message": f"XML parse error: {str(e)}"}
    try:
        with Device(host=device_ip, user=username, password=password, port=int(port)) as dev:
            result  = dev.rpc.get_config(
                filter_xml='<configuration><vlans/></configuration>',
                options={'format': 'text'}
            )
            evidence = result.text if hasattr(result, 'text') else str(result)
            if operation == "delete":
                confirmed = vlan_name not in evidence
                return {"status": "success" if confirmed else "failure", "operation": "delete",
                        "vlan_name": vlan_name, "evidence": evidence,
                        "message": f"Confirmed: {vlan_name} deleted" if confirmed else f"Audit failed: {vlan_name} still exists"}
            else:
                confirmed = vlan_name in evidence
                return {"status": "success" if confirmed else "failure", "operation": "create",
                        "vlan_name": vlan_name, "evidence": evidence,
                        "message": f"Confirmed: {vlan_name} present" if confirmed else f"Audit failed: {vlan_name} not found"}
    except Exception as e:
        return {"status": "failure", "operation": operation, "vlan_name": vlan_name,
                "evidence": "", "message": f"Audit error: {str(e)}"}


# ── Skill 6: get_inventory ───────────────────────────────────────
def get_device_inventory(device_ip: str, username: str, password: str,
                         port: str = "830") -> Dict[str, Any]:
    if not JUNOS_AVAILABLE:
        return {"status": "skipped", "vlans": [], "vlan_names": [], "raw_config": "",
                "message": "Junos modules not available"}
    try:
        with Device(host=device_ip, user=username, password=password, port=int(port)) as dev:
            result     = dev.rpc.get_config(
                filter_xml='<configuration><vlans/></configuration>',
                options={'format': 'text'}
            )
            raw_config = result.text if hasattr(result, 'text') else str(result)
            vlans, vlan_names = [], []
            lines = raw_config.split('\n')
            i = 0
            while i < len(lines):
                m = re.match(r'^    (\S+)\s*\{', lines[i])
                if m and m.group(1) not in ('vlans',):
                    name = m.group(1)
                    j, vid = i + 1, None
                    while j < len(lines) and '}' not in lines[j]:
                        mv = re.search(r'vlan-id\s+(\d+)', lines[j])
                        if mv:
                            vid = mv.group(1)
                        j += 1
                    if vid:
                        vlans.append({"name": name, "vlan_id": vid})
                        vlan_names.append(name)
                i += 1
            return {"status": "success", "vlans": vlans, "vlan_names": vlan_names,
                    "raw_config": raw_config, "message": f"Retrieved {len(vlans)} VLANs from {device_ip}"}
    except Exception as e:
        return {"status": "failure", "vlans": [], "vlan_names": [], "raw_config": "",
                "message": f"Inventory failed: {str(e)}"}


# ── Skill 7: lookup_documentation ───────────────────────────────
def lookup_documentation(query: str, retriever=None, top_k: int = 3) -> Dict[str, Any]:
    if retriever is None:
        return {"status": "failure", "documents": [], "context": "", "message": "Retriever not available"}
    try:
        docs      = retriever.invoke(query)
        documents = [doc.page_content for doc in docs[:top_k]]
        context   = "\n\n---\n\n".join(documents)
        return {"status": "success", "documents": documents, "context": context,
                "message": f"Found {len(documents)} documents for: {query}"}
    except Exception as e:
        return {"status": "failure", "documents": [], "context": "",
                "message": f"Lookup failed: {str(e)}"}


# ── Phase 6 Skills ──────────────────────────────────────────────

TASK_DECOMPOSER_PROMPT = """
You are a network operation task decomposer.
Analyze the user's network configuration request and break it down into atomic tasks.

[OUTPUT FORMAT]
Return ONLY valid JSON. No explanation. No markdown code blocks.
Schema:
{
  "tasks": [
    {
      "id": "task_1",
      "operation": "delete" | "create" | "configure_interface" | "skip",
      "target_vlan": "<VLAN name or null>",
      "vlan_id": "<VLAN ID or null>",
      "interface": "<interface name or null>",
      "description": "<one-line description>",
      "depends_on": ["task_N", ...]
    }
  ]
}

[DEPENDENCY RULES]
- Interface configuration on a VLAN depends on that VLAN's create task.
- VLAN delete depends on any interface task that removes that VLAN first.
- Independent tasks have empty depends_on: [].
- Never create circular dependencies.

[CURRENT NETWORK STATE]
{inventory_section}

[SMART DECISION RULES]
- If asked to DELETE a VLAN that does NOT exist in current state: set operation to "skip".
- If asked to CREATE a VLAN that ALREADY exists: set operation to "skip".

[EXAMPLES]
Input: "Delete SALES_VLAN ID 70 and add DEV_VLAN ID 200"
Output: {"tasks": [
  {"id":"task_1","operation":"delete","target_vlan":"SALES_VLAN","vlan_id":null,"interface":null,"description":"Delete SALES_VLAN","depends_on":[]},
  {"id":"task_2","operation":"create","target_vlan":"DEV_VLAN","vlan_id":"200","interface":null,"description":"Create DEV_VLAN","depends_on":[]}
]}
"""

def decompose_tasks(user_query: str, llm=None, inventory: Dict = None) -> Dict[str, Any]:
    if llm is None:
        return {"status": "failure", "tasks": [], "raw_response": "", "message": "LLM not provided"}
    try:
        if inventory and inventory.get("status") == "success":
            vlan_names = inventory.get("vlan_names", [])
            raw_cfg    = inventory.get("raw_config", "").strip()
            inv_section = (
                f"Existing VLANs on device: {vlan_names if vlan_names else '(none)'}\n"
                f"Raw config:\n{raw_cfg if raw_cfg else '(empty)'}"
            )
        else:
            inv_section = "(Inventory not available)"
        prompt_text = TASK_DECOMPOSER_PROMPT.replace("{inventory_section}", inv_section)
        prompt      = f"{prompt_text}\n\nInput: {user_query}\nOutput:"
        response    = llm.invoke(prompt)
        raw         = response.content.strip()
        json_match  = re.search(r"```json\s*(.+?)\s*```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        parsed = json.loads(raw)
        tasks  = parsed.get("tasks", [])
        return {"status": "success", "tasks": tasks, "raw_response": raw,
                "message": f"Decomposed into {len(tasks)} task(s)"}
    except json.JSONDecodeError as e:
        return {"status": "failure", "tasks": [], "raw_response": "",
                "message": f"JSON parse error: {e}"}
    except Exception as e:
        return {"status": "failure", "tasks": [], "raw_response": "",
                "message": f"task_decomposer error: {e}"}


def resolve_dependencies(tasks: List[Dict]) -> Dict[str, Any]:
    if not tasks:
        return {"status": "success", "execution_order": [], "message": "No tasks to resolve"}
    from collections import deque
    task_map = {t["id"]: t for t in tasks}
    task_ids = set(task_map.keys())
    for t in tasks:
        for dep in t.get("depends_on", []):
            if dep not in task_ids:
                return {"status": "error", "execution_order": [],
                        "message": f"Unknown dependency: '{t['id']}' depends on '{dep}'"}
    in_degree  = {t["id"]: 0 for t in tasks}
    dependents = {t["id"]: [] for t in tasks}
    for t in tasks:
        for dep in t.get("depends_on", []):
            in_degree[t["id"]] += 1
            dependents[dep].append(t["id"])
    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    execution_order = []
    while queue:
        tid  = queue.popleft()
        task = copy.deepcopy(task_map[tid])
        task["parallel"] = (
            len(task.get("depends_on", [])) == 0 and
            sum(1 for t in tasks if len(t.get("depends_on", [])) == 0) > 1
        )
        execution_order.append(task)
        for dep_id in dependents[tid]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                queue.append(dep_id)
    if len(execution_order) != len(tasks):
        remaining = [tid for tid, deg in in_degree.items() if deg > 0]
        return {"status": "error", "execution_order": [],
                "message": f"Circular dependency detected: {remaining}"}
    return {"status": "success", "execution_order": execution_order,
            "message": f"Resolved {len(execution_order)} task(s)"}


def aggregate_results(task_results: List[Dict]) -> Dict[str, Any]:
    succeeded, failed, skipped = [], [], []
    report_lines = ["=" * 60, "Phase 6/7 Orchestrator Execution Report", "=" * 60]
    for entry in task_results:
        task   = entry.get("task", {})
        result = entry.get("result", {})
        tid    = task.get("id", "unknown")
        op     = task.get("operation", "?")
        vlan   = task.get("target_vlan", "?")
        iface  = task.get("interface", "")
        desc   = task.get("description", "")
        deploy = result.get("deployment_status", {}) or {}
        audit  = result.get("audit_status") or {}
        deploy_status = deploy.get("status", "unknown")
        audit_status  = audit.get("status", "") if audit else ""
        label  = f"[{tid}] {op} / {vlan}" + (f" / {iface}" if iface else "")
        report_lines += [
            f"  {label}",
            f"  Desc: {desc}",
            f"  Valid: {'OK' if result.get('validation_status') else 'FAIL'}",
            f"  Deploy: {deploy_status}",
        ]
        if deploy.get("diff"):
            for line in deploy["diff"].strip().split("\n")[:5]:
                report_lines.append(f"  diff: {line}")
        if audit_status:
            report_lines.append(f"  Audit: {audit_status} - {audit.get('message','')}")
        if result.get("rollback_status"):
            rb = result["rollback_status"]
            report_lines.append(f"  Rollback: {rb.get('status','')} [{rb.get('mode','')}]")
        if deploy_status in ("success", "no_changes") and (not audit_status or audit_status == "success"):
            succeeded.append(tid)
            report_lines.append("  → SUCCESS\n")
        elif deploy_status == "skipped":
            skipped.append(tid)
            report_lines.append("  → SKIPPED\n")
        else:
            failed.append(tid)
            report_lines.append("  → FAILED\n")
    report_lines.append("=" * 60)
    if not failed and not skipped:
        overall = "all_success"
        summary = f"All {len(succeeded)} task(s) completed successfully"
    elif not failed and not succeeded and skipped:
        overall = "dry_run_complete"
        summary = f"Dry run complete: {len(skipped)} task(s) validated (deploy=False)"
    elif failed and not succeeded:
        overall = "all_failure"
        summary = f"All {len(failed)} task(s) failed"
    else:
        overall = "partial_failure"
        summary = f"{len(succeeded)} succeeded / {len(failed)} failed / {len(skipped)} skipped"
        if failed:
            summary += f" — Failed: {failed}"
    report_lines.append(f"  {summary}")
    report_lines.append("=" * 60)
    return {"status": overall, "summary": summary, "succeeded_tasks": succeeded,
            "failed_tasks": failed, "skipped_tasks": skipped, "report_lines": report_lines}


# ── Register all Skills ──────────────────────────────────────────
validate_xml_skill         = Skill("validate_xml", "Validate Junos XML structure", validate_xml_structure)
fix_xml_skill              = Skill("fix_xml", "Auto-fix common Junos XML errors", fix_xml_structure)
deploy_skill               = Skill("deploy_netconf", "Deploy XML to Junos via NETCONF", deploy_netconf_config)
rollback_skill             = Skill("rollback", "Rollback Junos config", rollback_config)
audit_skill                = Skill("audit", "Verify deployment via NETCONF", audit_deployment)
get_inventory_skill        = Skill("get_inventory", "Fetch VLAN list from device", get_device_inventory)
lookup_documentation_skill = Skill("lookup_documentation", "Search RAG knowledge base", lookup_documentation)
task_decomposer_skill      = Skill("task_decomposer", "Decompose NL request into task list", decompose_tasks)
dependency_resolver_skill  = Skill("dependency_resolver", "Resolve DAG dependencies", resolve_dependencies)
result_aggregator_skill    = Skill("result_aggregator", "Aggregate results into report", aggregate_results)

ALL_SKILLS = [
    validate_xml_skill, fix_xml_skill, deploy_skill,
    rollback_skill, audit_skill, get_inventory_skill, lookup_documentation_skill,
]
ALL_SKILLS_V6 = ALL_SKILLS + [task_decomposer_skill, dependency_resolver_skill, result_aggregator_skill]

# ─────────────────────────────────────────────────────────────────
# 3. MAF Reviewer Instructions (Phase 6 — intent-only)
# ─────────────────────────────────────────────────────────────────

REVIEWER_INSTRUCTIONS_V6 = """
You are an XML intent validator — NOT a Junos syntax expert.

[YOUR ONLY JOB]
Check whether the generated XML matches the user's stated intent.
Do NOT apply Junos-specific rules. Technical correctness is handled by validate_xml skill.

[THREE CHECKS ONLY]
1. Operation type match: delete intent → operation="delete"; create intent → no delete attribute
2. VLAN name match: VLAN name in XML must match the target from the user request
3. Basic XML parsability

[WHAT YOU MUST IGNORE]
- Whether <vlan-id> is present or absent
- Whether <vlans> wrapper exists
- Any other Junos-specific structural rules

[OUTPUT FORMAT]
First word MUST be exactly: APPROVE, IMPROVE, or REJECT
APPROVE: Intent matches.
IMPROVE: <specific intent mismatch only>
REJECT: <critical intent mismatch>
"""

# ─────────────────────────────────────────────────────────────────
# 4. Worker: NetconfRagWorkflow (Phase 5 / MAF)
# ─────────────────────────────────────────────────────────────────

class NetconfRagWorkflow:
    """
    MAF-based NETCONF RAG Worker.
    Steps: translate → inventory → RAG → Generator → Reviewer → validate → fix → deploy → audit → rollback
    """

    def __init__(self, retriever, llm, skills=None, max_retries=3, max_review_rounds=2, log_callback=None):
        self.retriever         = retriever
        self.llm               = llm
        self.max_retries       = max_retries
        self.max_review_rounds = max_review_rounds
        self.logs              = []
        self.conversation_history: List[Message] = []
        self.skill_execution_log: List[Dict]     = []
        self.log_callback      = log_callback  # callable(str) for live UI updates

        self.skills: Dict[str, Skill] = {}
        for sk in (skills or ALL_SKILLS):
            self.skills[sk.name] = sk

        self._initialize_agents()

    def _initialize_agents(self):
        generator_instructions = """
You are a dedicated JUNOS NETCONF XML generator.
Your ONLY task is to produce valid JUNOS XML.

[CRITICAL RULES]
1. OUTPUT ONLY RAW XML.
2. DO NOT include any introductory text, explanations.
3. DO NOT use markdown code blocks (no ```xml).
4. Start immediately with <configuration> and end with </configuration>.
5. If you cannot generate XML, output ONLY <configuration/>.

[CRITICAL TAG RULE]
- VLAN name tag MUST be <name>...</name>
- NEVER use numeric-only XML tags

[GOOD EXAMPLE - DELETE]
<vlan operation="delete"><name>VLAN_NAME</name></vlan>

[GOOD EXAMPLE - CREATE]
<vlan><name>VLAN_NAME</name><vlan-id>70</vlan-id></vlan>

[STRUCTURE EXAMPLE]
<configuration>
    <vlans>
        <vlan operation="delete">
            <name>SALES_VLAN</name>
        </vlan>
    </vlans>
</configuration>
"""
        self.xml_generator = Agent(
            name="XMLGenerator",
            client=make_client(DEFAULT_MODEL),
            instructions=generator_instructions
        )
        self.xml_reviewer = Agent(
            name="XMLReviewer",
            client=make_client(DEFAULT_MODEL),
            instructions=REVIEWER_INSTRUCTIONS_V6
        )

    def log(self, message: str):
        self.logs.append(message)
        if self.log_callback:
            self.log_callback(message)

    def add_message(self, role: str, text: str):
        self.conversation_history.append(Message(role=role, contents=[text]))

    def _extract_response_text(self, response) -> str:
        if hasattr(response, 'text'):
            return str(response.text)
        if hasattr(response, 'messages') and response.messages:
            for msg in response.messages:
                if hasattr(msg, 'text'):
                    return str(msg.text)
        return str(response)

    def _extract_xml(self, response: str) -> str:
        m = re.search(r'```xml\s*(<configuration>.*?</configuration>)\s*```', response, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'(<configuration>.*?</configuration>)', response, re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    def _run_skill(self, skill_name: str, **kwargs) -> Any:
        if skill_name == "lookup_documentation" and "retriever" not in kwargs:
            kwargs["retriever"] = self.retriever
        if skill_name not in self.skills:
            self.log(f"  [Skill] Unknown skill: {skill_name}")
            return None
        skill = self.skills[skill_name]
        self.log(f"  [Skill:{skill_name}] Running...")
        try:
            result = skill.execute(**kwargs)
            self.skill_execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "skill": skill_name,
                "result_summary": str(result)[:200]
            })
            self.log(f"  [Skill:{skill_name}] Done")
            return result
        except Exception as e:
            self.log(f"  [Skill:{skill_name}] Error: {e}")
            return None

    async def _run_skill_loop(self, xml_config: str, translated_query: str,
                               device_info: Optional[Dict] = None, deploy: bool = False) -> Dict[str, Any]:
        self.log("\n" + "=" * 60)
        self.log("Skill Loop — Phase: validate → fix → deploy → audit")
        self.log("=" * 60)
        current_xml = xml_config
        skill_steps = []

        # A: validate
        self.log("  [Step A] validate_xml")
        skill_steps.append("validate")
        val = self._run_skill("validate_xml", xml_config=current_xml)
        if val is None:
            return {"final_xml": current_xml, "valid": False,
                    "deployment_status": {"status": "skipped", "diff": "", "message": "validate_xml failed"},
                    "skill_steps": skill_steps}
        self.log(f"  Validation: valid={val['valid']} errors={val['errors']}")

        # B: fix if needed
        if not val['valid']:
            for attempt in range(3):
                self.log(f"  [Step B] fix_xml (attempt {attempt+1}/3)")
                skill_steps.append("fix")
                fix = self._run_skill("fix_xml", xml_config=current_xml, translated_query=translated_query)
                if not fix or not fix['success']:
                    self.log("  fix_xml failed")
                    break
                current_xml = fix['fixed_xml']
                self.log(f"  Fix applied: {fix['changes']}")
                skill_steps.append("re-validate")
                val = self._run_skill("validate_xml", xml_config=current_xml)
                if val and val['valid']:
                    self.log("  Re-validation passed")
                    break

        if not val or not val['valid']:
            self.log("  Final validation failed")
            return {"final_xml": current_xml, "valid": False,
                    "deployment_status": {"status": "skipped", "diff": "", "message": "Validation failed"},
                    "skill_steps": skill_steps}

        self.log("  XML validated — proceeding to deploy decision")

        if deploy and device_info:
            self.log("  [Step D] deploy_netconf")
            skill_steps.append("deploy")
            dep = self._run_skill("deploy_netconf",
                                   xml_config=current_xml, device_ip=device_info['ip'],
                                   username=device_info['username'], password=device_info['password'],
                                   port=device_info.get('port', '830'))
            if not dep or dep['status'] == "failure":
                err = dep.get('message', 'unknown') if dep else 'Skill error'
                self.log(f"  Deploy failed: {err}")
                skill_steps.append("rollback(candidate)")
                self._run_skill("rollback", device_ip=device_info['ip'],
                                username=device_info['username'], password=device_info['password'],
                                port=device_info.get('port', '830'), mode="candidate")
                return {"final_xml": current_xml, "valid": True,
                        "deployment_status": dep or {"status": "failure", "diff": "", "message": "Deploy failed"},
                        "skill_steps": skill_steps}
            self.log(f"  Deploy: {dep['status']}")
            if dep.get('diff'):
                self.log(f"  Diff:\n{dep['diff']}")

            # E: audit
            self.log("  [Step E] audit")
            skill_steps.append("audit")
            audit = self._run_skill("audit", xml_config=current_xml, device_ip=device_info['ip'],
                                    username=device_info['username'], password=device_info['password'],
                                    port=device_info.get('port', '830'))
            if audit:
                self.log(f"  Audit: {audit['status']} — {audit['message']}")
            if not audit or audit['status'] == "failure":
                skill_steps.append("rollback(rescue)")
                rb = self._run_skill("rollback", device_ip=device_info['ip'],
                                     username=device_info['username'], password=device_info['password'],
                                     port=device_info.get('port', '830'), mode="rescue")
                return {"final_xml": current_xml, "valid": True, "deployment_status": dep,
                        "audit_status": audit, "rollback_status": rb, "skill_steps": skill_steps}
            return {"final_xml": current_xml, "valid": True, "deployment_status": dep,
                    "audit_status": audit, "rollback_status": None, "skill_steps": skill_steps}
        else:
            reason = "deploy=False" if not deploy else "device_info missing"
            self.log(f"  Deploy skipped ({reason})")
            return {"final_xml": current_xml, "valid": True,
                    "deployment_status": {"status": "skipped", "diff": "", "message": reason},
                    "skill_steps": skill_steps}

    async def step1_translate_query(self, user_query: str) -> str:
        self.log("\n" + "=" * 60)
        self.log("Step 1: Translate query to English technical command")
        self.log("=" * 60)
        ascii_ratio = sum(1 for c in user_query if ord(c) < 128) / max(len(user_query), 1)
        if ascii_ratio >= 0.8:
            self.log(f"  English detected (ratio={ascii_ratio:.2f}) — skipping translation")
            self.add_message("user", user_query)
            self.add_message("system", f"Translation: (skipped) {user_query}")
            return user_query
        translation_prompt = (
            f"Convert the following network configuration request into a precise English technical command: "
            f"'{user_query}'. Output only the command text."
        )
        response       = self.llm.invoke(translation_prompt)
        translated     = response.content.strip()
        self.log(f"  Translated: {translated}")
        self.add_message("user", user_query)
        self.add_message("system", f"Translation: {translated}")
        return translated

    async def step2_retrieve_information(self, translated_query: str) -> List[str]:
        self.log("\n" + "=" * 60)
        self.log("Step 2: RAG retrieval")
        self.log("=" * 60)
        if not self.retriever:
            self.log("  Retriever unavailable")
            return []
        docs     = self.retriever.invoke(translated_query)
        contents = [doc.page_content for doc in docs]
        self.log(f"  Retrieved {len(contents)} document(s)")
        return contents

    async def step3_generate_and_review_xml(self, translated_query: str,
                                             retrieved_docs: List[str],
                                             inventory_info: Dict = None) -> tuple:
        self.log("\n" + "=" * 60)
        self.log("Step 3: Multi-Agent XML Generation + Review")
        self.log("=" * 60)
        context = "\n\n---\n\n".join(retrieved_docs)
        for attempt in range(self.max_retries):
            self.log(f"\n  Generation attempt {attempt+1}/{self.max_retries}")
            inv_section = ""
            if inventory_info and inventory_info.get('status') == 'success':
                vlan_names = inventory_info.get('vlan_names', [])
                raw_cfg    = inventory_info.get('raw_config', '').strip()
                inv_section = f"""
### Current Device State:
Existing VLANs: {vlan_names if vlan_names else '(none)'}
{raw_cfg if raw_cfg.strip() else '(empty)'}
"""
            gen_prompt = f"""
You are a Junos network engineer. Generate ONLY the JUNOS XML configuration.

[STRICT XML RULES]
- Output ONLY <configuration>...</configuration>. No explanations. No markdown.
- DELETE/REMOVE → operation="delete" attribute on <vlan> tag
- CREATE/ADD → no operation attribute, include <name> and <vlan-id>
- Use <name> tag for VLAN name
- DELETE: include ONLY <name> tag, NO <vlan-id>
{inv_section}
### Documentation Context:
{context}

### Request:
{translated_query}

Generate JUNOS XML now. ONLY XML. NO EXPLANATIONS.
"""
            self.log("  [Generator] Generating XML...")
            gen_response = await self.xml_generator.run(gen_prompt)
            raw_xml      = self._extract_response_text(gen_response)
            generated_xml = self._extract_xml(raw_xml)
            if not generated_xml:
                self.log("  [Generator] XML extraction failed")
                continue
            try:
                ET.fromstring(generated_xml)
            except ET.ParseError as e:
                self.log(f"  [Generator] XML parse error: {e}")
                continue
            self.log("  [Generator] XML generated successfully")
            self.add_message("assistant", f"[Generator] XML attempt {attempt+1}")

            for rr in range(self.max_review_rounds):
                self.log(f"  [Reviewer] Review round {rr+1}/{self.max_review_rounds}")
                inv_check = ""
                if inventory_info and inventory_info.get('status') == 'success':
                    vn = inventory_info.get('vlan_names', [])
                    inv_check = f"\nExisting VLANs on device: {vn}\n"
                review_prompt = f"""
Review this JUNOS XML for intent correctness:
{generated_xml}

User requirement: {translated_query}
{inv_check}
[CRITICAL JUNOS DELETE RULES]
DELETE: only <name> tag inside <vlan operation="delete">. NO <vlan-id>. This is CORRECT.

Check:
1. operation="delete" is an XML attribute on <vlan>, not a child tag
2. DELETE: ONLY <name> tag — APPROVE even if <vlan-id> is absent
3. CREATE: contains both <name> and <vlan-id>

IMPORTANT: Your response MUST start with exactly one of: APPROVE, IMPROVE, or REJECT.
"""
                review_response = await self.xml_reviewer.run(review_prompt)
                review_text     = self._extract_response_text(review_response)
                self.log(f"  [Reviewer] {review_text[:200]}")
                self.add_message("assistant", f"[Reviewer] {review_text[:100]}")

                if "APPROVE" in review_text.upper():
                    self.log("  [Reviewer] APPROVED")
                    return generated_xml, True
                elif "IMPROVE" in review_text.upper() and rr < self.max_review_rounds - 1:
                    improvement = review_text.split("IMPROVE:", 1)[-1].strip() if "IMPROVE:" in review_text else review_text
                    improve_prompt = f"""
[STRICT RULES]
- Output ONLY <configuration>...</configuration>.
Context: {context}
Request: {translated_query}
Previous XML: {generated_xml}
Reviewer Feedback: {improvement}
Improve the XML now.
"""
                    imp_response = await self.xml_generator.run(improve_prompt)
                    imp_raw      = self._extract_response_text(imp_response)
                    imp_xml      = self._extract_xml(imp_raw)
                    if imp_xml:
                        try:
                            ET.fromstring(imp_xml)
                            generated_xml = imp_xml
                            self.log("  [Generator] Improved XML generated")
                            continue
                        except ET.ParseError:
                            pass
                    self.log("  [Generator] Improvement failed — using original")
                    return generated_xml, True
                elif "REJECT" in review_text.upper():
                    self.log("  [Reviewer] REJECTED")
                    break
                else:
                    self.log("  [Reviewer] Unknown response — treating as APPROVE")
                    return generated_xml, True
        self.log(f"  XML generation failed after {self.max_retries} attempts")
        return "", False

    async def run(self, user_query: str, device_ip: str = None, username: str = None,
                  password: str = None, port: str = "830", deploy: bool = False) -> Dict[str, Any]:
        self.logs = []
        self.conversation_history = []
        self.skill_execution_log  = []

        self.log("\n" + "=" * 60)
        self.log("MAF NETCONF RAG Workflow — Starting")
        self.log(f"Max retries: {self.max_retries}  |  Review rounds: {self.max_review_rounds}")
        self.log("=" * 60)
        self.log(f"Query: {user_query}")

        result = {
            'user_query': user_query, 'translated_query': '',
            'retrieved_documents': [], 'generated_xml': '', 'final_xml': '',
            'validation_status': False, 'deployment_status': {},
            'skill_steps': [], 'skill_execution_log': [], 'conversation_history': []
        }

        try:
            result['translated_query'] = await self.step1_translate_query(user_query)
            inventory_info = None
            if all([device_ip, username, password]):
                self.log("\n" + "=" * 60)
                self.log("Step 0: Device Inventory")
                self.log("=" * 60)
                inv = self._run_skill("get_inventory", device_ip=device_ip,
                                      username=username, password=password, port=port)
                if inv and inv['status'] == 'success':
                    inventory_info = inv
                    self.log(f"  Current VLANs: {inv.get('vlan_names', [])}")
                    result['inventory'] = inv

            result['retrieved_documents'] = await self.step2_retrieve_information(result['translated_query'])
            raw_xml, review_passed = await self.step3_generate_and_review_xml(
                result['translated_query'], result['retrieved_documents'], inventory_info=inventory_info
            )
            result['generated_xml'] = raw_xml

            if not review_passed or not raw_xml:
                self.log("  XML generation failed — aborting workflow")
                result['validation_status'] = False
                return result

            device_info = None
            if deploy and all([device_ip, username, password]):
                device_info = {'ip': device_ip, 'username': username, 'password': password, 'port': port}
            elif deploy:
                self.log("  Device info incomplete — deploy skipped")

            skill_result = await self._run_skill_loop(
                xml_config=raw_xml, translated_query=result['translated_query'],
                device_info=device_info, deploy=deploy
            )
            result['final_xml']           = skill_result['final_xml']
            result['validation_status']   = skill_result['valid']
            result['deployment_status']   = skill_result['deployment_status']
            result['audit_status']        = skill_result.get('audit_status')
            result['rollback_status']     = skill_result.get('rollback_status')
            result['skill_steps']         = skill_result['skill_steps']
            result['skill_execution_log'] = self.skill_execution_log
            result['conversation_history'] = [
                {"role": msg.role, "text": msg.text if hasattr(msg, 'text') else str(msg.contents)}
                for msg in self.conversation_history
            ]
            self.log("\n" + "=" * 60)
            self.log("Workflow Complete")
            self.log(f"Skill steps: {' → '.join(skill_result['skill_steps'])}")
            self.log("=" * 60)
        except Exception as e:
            self.log(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        return result


# ─────────────────────────────────────────────────────────────────
# 5. Orchestrator (Phase 6/7)
# ─────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Phase 6 Orchestrator-Worker pattern.
    task_decomposer → dependency_resolver → Worker(s) → result_aggregator
    """

    def __init__(self, retriever, llm, skills=None, max_retries=3, max_review_rounds=2, log_callback=None):
        self.retriever         = retriever
        self.llm               = llm
        self.max_retries       = max_retries
        self.max_review_rounds = max_review_rounds
        self.logs              = []
        self.log_callback      = log_callback

        self.skills: Dict[str, Skill] = {}
        for sk in (skills or ALL_SKILLS_V6):
            self.skills[sk.name] = sk

    def log(self, message: str):
        self.logs.append(message)
        if self.log_callback:
            self.log_callback(message)

    def _run_skill(self, skill_name: str, **kwargs):
        if skill_name not in self.skills:
            self.log(f"  [Skill] Unknown: {skill_name}")
            return None
        self.log(f"  [Skill:{skill_name}] Running...")
        try:
            result = self.skills[skill_name].execute(**kwargs)
            self.log(f"  [Skill:{skill_name}] Done")
            return result
        except Exception as e:
            self.log(f"  [Skill:{skill_name}] Error: {e}")
            return None

    def _build_worker_query(self, task: Dict) -> str:
        op    = task.get("operation", "")
        vlan  = task.get("target_vlan", "")
        vid   = task.get("vlan_id", "")
        iface = task.get("interface", "")
        desc  = task.get("description", "")
        if op == "delete":
            return f"Delete VLAN named {vlan}."
        elif op == "create":
            vid_part = f" with VLAN ID {vid}" if vid else ""
            return f"Create VLAN named {vlan}{vid_part}."
        elif op == "configure_interface":
            return f"Configure interface {iface} to use VLAN {vlan}."
        return desc or f"{op} {vlan}"

    async def run(self, user_query: str, device_ip: str = None, username: str = None,
                  password: str = None, port: str = "830", deploy: bool = False) -> Dict[str, Any]:
        self.logs = []
        self.log("\n" + "=" * 70)
        self.log("Orchestrator (Phase 6) — Starting")
        self.log("=" * 70)
        self.log(f"Query: {user_query}")

        result = {
            "user_query": user_query, "tasks": [], "execution_order": [],
            "task_results": [], "aggregated": {}, "orchestrator_logs": self.logs
        }

        # Step 0: get_inventory
        orchestrator_inventory = None
        if all([device_ip, username, password]):
            self.log("\n" + "-" * 50)
            self.log("[Step 0] get_inventory — Orchestrator-level device state")
            inv = self._run_skill("get_inventory", device_ip=device_ip,
                                  username=username, password=password, port=port)
            if inv and inv.get("status") == "success":
                orchestrator_inventory = inv
                self.log(f"  Current VLANs: {inv.get('vlan_names', [])}")
                result["orchestrator_inventory"] = inv
            else:
                self.log(f"  Inventory failed (continuing): {(inv or {}).get('message','')}")

        # Step 1: task_decomposer
        self.log("\n" + "-" * 50)
        self.log("[Step 1] task_decomposer")
        decompose = self._run_skill("task_decomposer", user_query=user_query,
                                    llm=self.llm, inventory=orchestrator_inventory)
        if not decompose or decompose["status"] != "success":
            msg = (decompose or {}).get("message", "Skill error")
            self.log(f"  task_decomposer FAILED: {msg}")
            result["aggregated"] = {"status": "all_failure", "summary": f"task_decomposer failed: {msg}", "report_lines": []}
            return result

        tasks = decompose["tasks"]
        result["tasks"] = tasks
        self.log(f"  Detected {len(tasks)} task(s)")
        for t in tasks:
            dep_str = f" (depends_on: {t.get('depends_on', [])})" if t.get("depends_on") else ""
            self.log(f"    {t['id']}: {t.get('operation','?')} / {t.get('target_vlan','?')}{dep_str}")

        # Step 2: dependency_resolver
        self.log("\n" + "-" * 50)
        self.log("[Step 2] dependency_resolver")
        resolve = self._run_skill("dependency_resolver", tasks=tasks)
        if not resolve or resolve["status"] != "success":
            msg = (resolve or {}).get("message", "Skill error")
            self.log(f"  dependency_resolver FAILED: {msg}")
            result["aggregated"] = {"status": "all_failure", "summary": f"dependency_resolver failed: {msg}", "report_lines": []}
            return result

        execution_order = resolve["execution_order"]
        result["execution_order"] = execution_order
        self.log(f"  Execution order: {[t['id'] for t in execution_order]}")
        parallel = [t["id"] for t in execution_order if t.get("parallel")]
        if parallel:
            self.log(f"  Parallel-capable tasks: {parallel}")

        # Step 3: dispatch Workers
        self.log("\n" + "-" * 50)
        self.log(f"[Step 3] Dispatching {len(execution_order)} task(s) to Workers")
        task_results = []

        for idx, task in enumerate(execution_order, 1):
            tid = task["id"]
            self.log(f"\n  [{idx}/{len(execution_order)}] Worker: {tid}")
            self.log(f"    operation={task.get('operation','?')} | vlan={task.get('target_vlan','?')} | iface={task.get('interface','-')}")

            if task.get("operation") == "skip":
                self.log(f"  SKIP: {tid} — {task.get('description','')}")
                task_results.append({
                    "task": task,
                    "result": {
                        "validation_status": True,
                        "deployment_status": {"status": "no_changes", "diff": "",
                                              "message": f"Skipped: {task.get('description','')}"},
                        "audit_status": None, "rollback_status": None
                    }
                })
                continue

            worker_query = self._build_worker_query(task)
            self.log(f"    Worker query: {worker_query}")

            worker = NetconfRagWorkflow(
                retriever=self.retriever, llm=self.llm, skills=ALL_SKILLS,
                max_retries=self.max_retries, max_review_rounds=self.max_review_rounds,
                log_callback=self.log_callback
            )
            worker.xml_reviewer.instructions = REVIEWER_INSTRUCTIONS_V6

            try:
                worker_result = await worker.run(
                    user_query=worker_query, device_ip=device_ip, username=username,
                    password=password, port=port, deploy=deploy
                )
            except Exception as e:
                self.log(f"  Worker error: {e}")
                worker_result = {
                    "validation_status": False,
                    "deployment_status": {"status": "failure", "diff": "", "message": str(e)},
                    "audit_status": None, "rollback_status": None
                }

            dep_status = worker_result.get("deployment_status", {})
            dep_ok     = dep_status.get("status") in ("success", "no_changes", "skipped")
            audit_ok   = (worker_result.get("audit_status") or {}).get("status", "success") in ("success", "skipped", "")
            success    = dep_ok and audit_ok

            self.log(f"  {'OK' if success else 'FAIL'} {tid} → deploy={dep_status.get('status','?')}")
            task_results.append({"task": task, "result": worker_result})

            if not success:
                self.log(f"  Task {tid} failed — aborting remaining tasks")
                for rem in execution_order[idx:]:
                    task_results.append({
                        "task": rem,
                        "result": {"validation_status": False,
                                   "deployment_status": {"status": "skipped", "diff": "",
                                                         "message": f"Skipped due to failure of {tid}"},
                                   "audit_status": None, "rollback_status": None}
                    })
                break

        result["task_results"] = task_results

        # Step 4: result_aggregator
        self.log("\n" + "-" * 50)
        self.log("[Step 4] result_aggregator")
        aggregated = self._run_skill("result_aggregator", task_results=task_results)
        result["aggregated"] = aggregated or {}
        if aggregated:
            self.log(f"\n{aggregated['summary']}")
            for line in aggregated.get("report_lines", []):
                self.log(line)

        self.log("\n" + "=" * 70)
        self.log("Orchestrator — Complete")
        self.log("=" * 70)
        return result


# ─────────────────────────────────────────────────────────────────
# 6. NiceGUI UI
# ─────────────────────────────────────────────────────────────────

@ui.page('/')
def index():
    ui.navigate.to('/netconf_rag')
@ui.page('/netconf_rag')
def main_page():
    ui.page_title('MAF NETCONF RAG Agent')

    # ── Header ───────────────────────────────────────────────────
    with ui.card().classes('w-full q-pa-lg bg-grey-9 shadow-2'):
        ui.label('MAF NETCONF × Agentic RAG') \
            .classes('text-h4 text-bold text-center text-white')
        ui.label('Orchestrator-Worker Pattern  |  Phase 7 — Policy + Safety + Audit') \
            .classes('text-subtitle1 text-center text-grey-4')

    ui.space()

    # ── Architecture diagram ──────────────────────────────────────
    with ui.expansion('Architecture Overview', icon='account_tree').classes('w-full q-mt-sm'):
        ui.markdown("""
```
User Query (EN/JP)
      │
      ▼
[Orchestrator]
  ├─ Step 0: get_inventory        ← Current device VLAN state
  ├─ Step 1: task_decomposer      ← NL → Task list (JSON / DAG)
  ├─ Step 2: dependency_resolver  ← Topological sort
  └─ Step 3: Worker(s) dispatch
       └─ [Worker — Phase 5]
            ├─ translate_query
            ├─ RAG retrieval (FAISS)
            ├─ Generator Agent (LLaMA 3.3 70B)
            ├─ Reviewer Agent  (LLaMA 3.3 70B)
            ├─ validate_xml Skill
            ├─ fix_xml Skill
            ├─ deploy_netconf Skill (NETCONF/830)
            ├─ audit Skill
            └─ rollback Skill (on failure)
  └─ Step 4: result_aggregator    ← Final report
```
        """)

    ui.separator()

    # ── Device connection ─────────────────────────────────────────
    with ui.row().classes('w-full gap-4'):
        with ui.card().classes('col-grow'):
            ui.label('Device Connection').classes('text-h6 text-bold')
            ui.label('Junos / NETCONF').classes('text-caption text-grey-6')
            global device_hostname, device_username, password_input, port_input
            device_hostname  = ui.input('Device IP',  value='172.20.100.21').props('outlined dense').classes('w-full')
            device_username  = ui.input('Username',    value='admin').props('outlined dense').classes('w-full')
            password_input   = ui.input('Password',    value='').props('outlined dense type=password').classes('w-full')
            port_input       = ui.input('NETCONF Port', value='830').props('outlined dense').classes('w-full')

        with ui.card().classes('col-grow'):
            ui.label('Run Options').classes('text-h6 text-bold')
            global deploy_toggle, orchestrator_toggle, max_retries_input
            deploy_toggle      = ui.switch('Deploy to Device', value=True)
            orchestrator_toggle = ui.switch('Use Orchestrator (multi-task)', value=True)
            max_retries_input  = ui.number('Max Retries', value=3, min=1, max=10).props('outlined dense')
            ui.label('Groq Model: llama-3.3-70b-versatile').classes('text-caption text-grey-6 q-mt-sm')

    ui.separator()

    # ── Query input ───────────────────────────────────────────────
    with ui.card().classes('w-full'):
        ui.label('Configuration Request').classes('text-h6 text-bold')
        ui.label('Enter your network change in English or Japanese').classes('text-caption text-grey-6')
        global query_input
        query_input = ui.textarea(
            label='e.g. "Add VLAN100 with ID 100, then delete SALES_VLAN"',
            value="Add 'DEV_VLAN' with VLAN ID 50."
        ).props('outlined rows=3').classes('w-full')

        with ui.row().classes('gap-2 q-mt-sm'):
            ui.button('Run Agent', icon='play_arrow',
                       on_click=lambda: asyncio.create_task(run_agent())).classes('bg-green-600 text-white')
            ui.button('Clear', icon='clear',
                       on_click=clear_ui).classes('bg-grey-600 text-white')

    ui.separator()

    # ── DAG visualization ─────────────────────────────────────────
    with ui.row().classes('w-full gap-4'):
        with ui.card().classes('col-4'):
            ui.label('Task DAG').classes('text-h6 text-bold')
            ui.label('Dependency graph generated by task_decomposer').classes('text-caption text-grey-6')
            global dag_container
            dag_container = ui.column().classes('w-full')
            with dag_container:
                ui.label('DAG will appear after query execution').classes('text-grey-5 text-caption')

        # ── Execution log ─────────────────────────────────────────
        with ui.card().classes('col-grow'):
            ui.label('Execution Log').classes('text-h6 text-bold')
            global exec_log
            exec_log = ui.log(max_lines=300).classes('w-full').style('height: 300px; font-size: 12px;')

    ui.separator()

    # ── Results ───────────────────────────────────────────────────
    with ui.row().classes('w-full gap-4'):
        with ui.card().classes('col-grow'):
            ui.label('Generated XML').classes('text-h6 text-bold')
            global xml_display
            xml_display = ui.textarea(placeholder='Generated XML config will appear here...') \
                .props('readonly outlined rows=15').classes('w-full').style('font-family: monospace; font-size: 12px;')

        with ui.card().classes('col-grow'):
            ui.label('Deployment Summary').classes('text-h6 text-bold')
            global result_display
            result_display = ui.textarea(placeholder='Deployment result and audit will appear here...') \
                .props('readonly outlined rows=15').classes('w-full').style('font-family: monospace; font-size: 12px;')

    # ── Skill execution log ───────────────────────────────────────
    with ui.expansion('Skill Execution Log', icon='construction').classes('w-full q-mt-sm'):
        global skill_log_display
        skill_log_display = ui.textarea(placeholder='Skill execution details...') \
            .props('readonly outlined rows=8').classes('w-full').style('font-family: monospace; font-size: 11px;')

    ui.separator()
    ui.label('maf-netconf-rag-gui | MAF Phase 7 | Groq LLaMA 3.3 70B') \
        .classes('text-caption text-grey-5 text-center q-mb-md')


def clear_ui():
    exec_log.clear()
    xml_display.value        = ""
    result_display.value     = ""
    skill_log_display.value  = ""
    dag_container.clear()
    with dag_container:
        ui.label('DAG will appear after query execution').classes('text-grey-5 text-caption')


def render_dag(tasks: List[Dict], execution_order: List[Dict]):
    dag_container.clear()
    if not tasks:
        with dag_container:
            ui.label('No tasks detected').classes('text-grey-5 text-caption')
        return

    op_colors = {
        "delete": "red-8", "create": "green-7",
        "configure_interface": "blue-7", "skip": "grey-5"
    }
    op_icons = {
        "delete": "delete", "create": "add_circle",
        "configure_interface": "settings_ethernet", "skip": "skip_next"
    }

    with dag_container:
        ui.label(f'{len(tasks)} Task(s) — Execution order: {" → ".join(t["id"] for t in execution_order)}') \
            .classes('text-caption text-grey-6 q-mb-xs')
        for task in execution_order:
            color = op_colors.get(task.get("operation", ""), "grey-7")
            icon  = op_icons.get(task.get("operation", ""), "task")
            with ui.card().classes(f'w-full q-pa-sm q-mb-xs bg-{color} text-white'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon(icon).classes('text-white')
                    ui.label(f"{task['id']}: {task.get('operation','?').upper()}").classes('text-bold text-sm')
                    if task.get('parallel'):
                        ui.badge('parallel', color='yellow-8')
                ui.label(f"Target: {task.get('target_vlan','?')}" +
                         (f" | Interface: {task.get('interface')}" if task.get('interface') else "") +
                         (f" | VLAN ID: {task.get('vlan_id')}" if task.get('vlan_id') else "")) \
                    .classes('text-xs')
                if task.get('depends_on'):
                    ui.label(f"Depends on: {', '.join(task['depends_on'])}").classes('text-xs text-yellow-3')
                ui.label(task.get('description', '')).classes('text-xs text-white-7')


def pretty_xml(xml_str: str) -> str:
    try:
        dom    = minidom.parseString(xml_str)
        pretty = dom.toprettyxml(indent="  ")
        return "\n".join(line for line in pretty.split("\n") if line.strip())
    except Exception:
        return xml_str


async def run_agent():
    clear_ui()
    query  = query_input.value.strip()
    if not query:
        exec_log.push("Error: Please enter a configuration request.")
        return

    exec_log.push("=" * 60)
    exec_log.push("MAF NETCONF RAG Agent — Starting")
    exec_log.push("=" * 60)

    host     = device_hostname.value.strip()
    user     = device_username.value.strip()
    pwd      = password_input.value.strip()
    prt      = port_input.value.strip() or "830"
    deploy   = deploy_toggle.value
    use_orch = orchestrator_toggle.value
    retries  = int(max_retries_input.value or 3)

    def log_to_ui(msg: str):
        exec_log.push(msg)

    if use_orch:
        # ── Orchestrator mode ─────────────────────────────────────
        exec_log.push("Mode: Orchestrator (multi-task DAG)")
        orch = OrchestratorAgent(
            retriever=retriever, llm=llm, skills=ALL_SKILLS_V6,
            max_retries=retries, max_review_rounds=2, log_callback=log_to_ui
        )
        orch_result = await orch.run(
            user_query=query, device_ip=host, username=user, password=pwd,
            port=prt, deploy=deploy
        )

        # Render DAG
        render_dag(orch_result.get("tasks", []), orch_result.get("execution_order", []))

        # Collect XMLs
        xml_parts = []
        for entry in orch_result.get("task_results", []):
            task   = entry.get("task", {})
            res    = entry.get("result", {})
            xml    = res.get("final_xml") or res.get("generated_xml", "")
            if xml:
                xml_parts.append(f"<!-- Task: {task.get('id','')} — {task.get('description','')} -->\n{pretty_xml(xml)}")
        xml_display.value = "\n\n".join(xml_parts) if xml_parts else "(No XML generated)"

        # Summary
        agg   = orch_result.get("aggregated", {})
        lines = [
            "=" * 50,
            "Orchestrator Result Summary",
            "=" * 50,
            f"Status: {agg.get('status', 'N/A')}",
            f"Summary: {agg.get('summary', 'N/A')}",
            "",
            f"Tasks detected   : {len(orch_result.get('tasks', []))}",
            f"Tasks executed   : {len(orch_result.get('task_results', []))}",
            f"Succeeded        : {len(agg.get('succeeded_tasks', []))}",
            f"Failed           : {len(agg.get('failed_tasks', []))}",
            f"Skipped          : {len(agg.get('skipped_tasks', []))}",
            "",
            "--- Task Details ---",
        ]
        for entry in orch_result.get("task_results", []):
            task    = entry.get("task", {})
            res     = entry.get("result", {})
            dep_s   = res.get("deployment_status", {})
            audit_s = res.get("audit_status") or {}
            lines += [
                f"[{task.get('id')}] {task.get('operation','?')} / {task.get('target_vlan','?')}",
                f"  Deploy : {dep_s.get('status','?')} — {dep_s.get('message','')}",
                f"  Audit  : {audit_s.get('status','-')} — {audit_s.get('message','')}",
                "",
            ]
        if agg.get("report_lines"):
            lines.append("--- Full Report ---")
            lines.extend(agg["report_lines"])
        result_display.value = "\n".join(lines)

        # Skill log
        skill_lines = []
        for entry in orch_result.get("task_results", []):
            res = entry.get("result", {})
            for sk_entry in res.get("skill_execution_log", []):
                skill_lines.append(f"[{sk_entry['timestamp']}] {sk_entry['skill']}: {sk_entry['result_summary']}")
        skill_log_display.value = "\n".join(skill_lines) if skill_lines else "(No skill log)"

    else:
        # ── Single Worker mode ────────────────────────────────────
        exec_log.push("Mode: Single Worker (direct, no Orchestrator)")
        worker = NetconfRagWorkflow(
            retriever=retriever, llm=llm, skills=ALL_SKILLS,
            max_retries=retries, max_review_rounds=2, log_callback=log_to_ui
        )
        w_result = await worker.run(
            user_query=query, device_ip=host, username=user, password=pwd,
            port=prt, deploy=deploy
        )

        # No DAG in single-worker mode
        dag_container.clear()
        with dag_container:
            ui.label('(Single-worker mode — no DAG)').classes('text-grey-5 text-caption')

        xml_display.value = pretty_xml(w_result.get("final_xml") or w_result.get("generated_xml", ""))

        dep_s   = w_result.get("deployment_status", {})
        audit_s = w_result.get("audit_status") or {}
        rb_s    = w_result.get("rollback_status") or {}
        lines   = [
            "=" * 50,
            "Single Worker Result",
            "=" * 50,
            f"Query      : {w_result.get('user_query','')}",
            f"Translated : {w_result.get('translated_query','')}",
            f"RAG docs   : {len(w_result.get('retrieved_documents', []))}",
            f"Valid      : {w_result.get('validation_status', False)}",
            f"Deploy     : {dep_s.get('status','?')} — {dep_s.get('message','')}",
        ]
        if dep_s.get("diff"):
            lines.append(f"Diff:\n{dep_s['diff']}")
        if audit_s:
            lines.append(f"Audit      : {audit_s.get('status','-')} — {audit_s.get('message','')}")
        if rb_s:
            lines.append(f"Rollback   : {rb_s.get('status','-')} [{rb_s.get('mode','')}]")
        lines.append(f"\nSkill steps: {' → '.join(w_result.get('skill_steps', []))}")
        result_display.value = "\n".join(lines)

        skill_lines = [
            f"[{e['timestamp']}] {e['skill']}: {e['result_summary']}"
            for e in w_result.get("skill_execution_log", [])
        ]
        skill_log_display.value = "\n".join(skill_lines) if skill_lines else "(No skill log)"

    exec_log.push("\n" + "=" * 60)
    exec_log.push("Agent run complete.")
    exec_log.push("=" * 60)


# ─────────────────────────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=8080,
        title="MAF NETCONF RAG Agent",
        favicon="🌐",
        dark=True,
        reload=False,
    )
