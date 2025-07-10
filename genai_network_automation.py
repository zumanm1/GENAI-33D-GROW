"""
GenAI Network Automation Tool
============================

Advanced network automation using NLP, LLM, and AI agents for Cisco device management.
Features:
- NLP to Cisco CLI conversion
- CrewAI agent validation and cleanup
- ChromaDB for command context
- Netmiko for device interaction
- SQLite for device inventory
- Real-time validation and deployment
"""

import os
import sqlite3
import re
import json
from typing import List, Tuple, Dict, Optional
import streamlit as st
import requests
from chromadb import Client as ChromaClient
from crewai import Agent, Task, Crew
from netmiko import ConnectHandler, NetmikoTimeoutException, NetmikoAuthenticationException
import time

# -------------------------------------------------
# Configuration
# -------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:latest")
DEVICES_DB = os.getenv("DEVICES_DB", "devices.db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cisco_cmds")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def run_ollama(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """Synchronous call to local Ollama REST API."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"System: {system}\n\nUser: {prompt}\n\nAssistant:",
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        st.error(f"Ollama API error: {str(e)}")
        return ""

def chroma_search(query: str, top_k: int = 4) -> str:
    """Return top-k Cisco docs snippets to give the LLM more context."""
    try:
        cc = ChromaClient()
        collection = cc.get_collection(CHROMA_COLLECTION)
        results = collection.query(query_texts=[query], n_results=top_k)
        return "\n".join(chunk for chunk in results["documents"][0]) if results["documents"] else ""
    except Exception as e:
        st.warning(f"ChromaDB search failed: {str(e)}")
        return ""

def fetch_devices() -> List[Tuple[int, str]]:
    """Fetch devices from SQLite database."""
    try:
        with sqlite3.connect(DEVICES_DB) as conn:
            cur = conn.execute("SELECT id, name FROM devices ORDER BY name")
            return cur.fetchall()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

def get_device(device_id: int) -> dict:
    """Get device configuration from database."""
    try:
        with sqlite3.connect(DEVICES_DB) as conn:
            cur = conn.execute(
                "SELECT host, username, password, device_type FROM devices WHERE id = ?",
                (device_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("Device not found in DB")
            host, username, password, device_type = row
            return {
                "device_type": device_type,
                "host": host,
                "username": username,
                "password": password,
                "fast_cli": False,
            }
    except Exception as e:
        st.error(f"Device configuration error: {str(e)}")
        return {}

def clean_with_agent(raw_cmds: str) -> str:
    """Use CrewAI to strip non-Cisco lines and duplicates."""
    try:
        validator = Agent(
            role="Cisco IOS Command Validator",
            goal="Return ONLY Cisco-compatible configuration lines, one per line, no explanations.",
            backstory=(
                "A veteran network engineer with deep knowledge of Cisco IOS/IOS-XE command syntax. "
                "Expert at cleaning and validating configuration commands."
            ),
            verbose=True,
            allow_delegation=False,
        )
        
        task = Task(
            description=(
                f"Clean and validate the following candidate configuration. "
                f"Remove any non-Cisco commands, duplicates, or invalid syntax. "
                f"Return only valid Cisco IOS configuration commands, one per line:\n\n{raw_cmds}"
            ),
            agent=validator,
            expected_output="Cleaned Cisco IOS configuration commands only, one per line.",
        )
        
        crew = Crew(agents=[validator], tasks=[task], verbose=True)
        result = crew.run()
        return result.strip()
    except Exception as e:
        st.error(f"CrewAI validation failed: {str(e)}")
        return raw_cmds

def split_commands(block: str) -> List[str]:
    """Split command block into individual commands."""
    return [line.strip() for line in block.splitlines() if line.strip()]

def netmiko_send_config(device: dict, commands: List[str]) -> str:
    """Send configuration commands to device."""
    try:
        with ConnectHandler(**device) as conn:
            output = conn.send_config_set(commands)
            try:
                output += "\n" + conn.save_config()
            except:
                pass  # Not all drivers implement save_config
            return output
    except Exception as e:
        raise Exception(f"Configuration deployment failed: {str(e)}")

def netmiko_send_commands(device: dict, commands: List[str]) -> str:
    """Send show/exec commands to device."""
    try:
        with ConnectHandler(**device) as conn:
            output = ""
            for cmd in commands:
                output += f"\n{cmd}\n" + conn.send_command(cmd)
            return output
    except Exception as e:
        raise Exception(f"Command execution failed: {str(e)}")

def validate_cisco_commands(commands: List[str]) -> Tuple[List[str], List[str]]:
    """Validate Cisco commands and return valid/invalid lists."""
    valid_commands = []
    invalid_commands = []
    
    # Basic Cisco command patterns
    cisco_patterns = [
        r'^interface\s+',
        r'^ip\s+',
        r'^router\s+',
        r'^access-list\s+',
        r'^route-map\s+',
        r'^vlan\s+',
        r'^spanning-tree\s+',
        r'^no\s+',
        r'^shutdown$',
        r'^no shutdown$',
        r'^description\s+',
        r'^hostname\s+',
        r'^banner\s+',
        r'^line\s+',
        r'^username\s+',
        r'^enable\s+',
        r'^service\s+',
        r'^logging\s+',
        r'^ntp\s+',
        r'^snmp-server\s+',
        r'^clock\s+',
        r'^timezone\s+',
        r'^crypto\s+',
        r'^aaa\s+',
        r'^radius-server\s+',
        r'^tacacs-server\s+',
        r'^key\s+',
        r'^crypto\s+',
        r'^ipv6\s+',
        r'^ospf\s+',
        r'^eigrp\s+',
        r'^bgp\s+',
        r'^isis\s+',
        r'^mpls\s+',
        r'^vrf\s+',
        r'^policy-map\s+',
        r'^class-map\s+',
        r'^queue-limit\s+',
        r'^bandwidth\s+',
        r'^delay\s+',
        r'^priority\s+',
        r'^dscp\s+',
        r'^precedence\s+',
        r'^cos\s+',
        r'^qos\s+',
        r'^mls\s+',
        r'^switchport\s+',
        r'^channel-group\s+',
        r'^port-channel\s+',
        r'^etherchannel\s+',
        r'^lacp\s+',
        r'^pagp\s+',
        r'^udld\s+',
        r'^loopback\s+',
        r'^tunnel\s+',
        r'^dialer\s+',
        r'^async\s+',
        r'^serial\s+',
        r'^atm\s+',
        r'^frame-relay\s+',
        r'^x25\s+',
        r'^isdn\s+',
        r'^voice\s+',
        r'^telephony-service\s+',
        r'^ephone\s+',
        r'^ephone-dn\s+',
        r'^sccp\s+',
        r'^h323\s+',
        r'^sip\s+',
        r'^mgcp\s+',
        r'^skinny\s+',
        r'^call-manager-fallback\s+',
        r'^call-manager\s+',
        r'^gatekeeper\s+',
        r'^gateway\s+',
        r'^trunk\s+',
        r'^linecode\s+',
        r'^framing\s+',
        r'^cablelength\s+',
        r'^clock\s+source\s+',
        r'^encapsulation\s+',
        r'^keepalive\s+',
        r'^carrier-delay\s+',
        r'^cdp\s+',
        r'^lldp\s+',
        r'^udld\s+',
        r'^vtp\s+',
        r'^vtp\s+mode\s+',
        r'^vtp\s+domain\s+',
        r'^vtp\s+password\s+',
        r'^vtp\s+version\s+',
        r'^vtp\s+pruning\s+',
        r'^vtp\s+tracking\s+',
        r'^vtp\s+file\s+',
        r'^vtp\s+interface\s+',
        r'^vtp\s+primary\s+',
        r'^vtp\s+secondary\s+',
        r'^vtp\s+client\s+',
        r'^vtp\s+server\s+',
        r'^vtp\s+transparent\s+',
        r'^vtp\s+off\s+',
        r'^vtp\s+version\s+1\s+',
        r'^vtp\s+version\s+2\s+',
        r'^vtp\s+version\s+3\s+',
        r'^vtp\s+version\s+4\s+',
        r'^vtp\s+version\s+5\s+',
        r'^vtp\s+version\s+6\s+',
        r'^vtp\s+version\s+7\s+',
        r'^vtp\s+version\s+8\s+',
        r'^vtp\s+version\s+9\s+',
        r'^vtp\s+version\s+10\s+',
    ]
    
    for cmd in commands:
        cmd_lower = cmd.lower()
        is_valid = False
        
        for pattern in cisco_patterns:
            if re.match(pattern, cmd_lower):
                is_valid = True
                break
        
        if is_valid:
            valid_commands.append(cmd)
        else:
            invalid_commands.append(cmd)
    
    return valid_commands, invalid_commands

def generate_validation_commands(intent: str, deployed_commands: List[str]) -> List[str]:
    """Generate validation commands based on intent and deployed commands."""
    try:
        validation_prompt = f"""
        Generate Cisco show/exec commands to validate the following configuration:
        
        Intent: {intent}
        Deployed Commands: {deployed_commands}
        
        Provide 3-5 show commands that would verify the configuration was applied correctly.
        Return only the commands, one per line, no explanations.
        """
        
        validation_commands = run_ollama(
            prompt=validation_prompt,
            system="You are a Cisco network engineer. Generate show commands to validate configuration changes.",
            temperature=0.1
        )
        
        return split_commands(validation_commands)
    except Exception as e:
        st.error(f"Validation command generation failed: {str(e)}")
        return []

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

def show_genai_network_automation():
    """Main GenAI Network Automation interface."""
    
    st.markdown('<div class="main-header"><h1>🧠⚡ GenAI Network Automation</h1><p>NLP-powered Cisco device configuration and management</p></div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h3>🎯 Operation Mode</h3></div>', unsafe_allow_html=True)
        
        mode = st.radio(
            "Select Mode",
            ("🚀 Push Configuration", "📊 Retrieve Information", "🔍 Validate Configuration"),
            format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
        )
        
        # Device selection
        st.markdown('<div class="feature-card"><h3>📱 Target Device</h3></div>', unsafe_allow_html=True)
        
        device_choices = fetch_devices()
        if not device_choices:
            st.error("❌ No devices found in database")
            st.info("💡 Add devices to the database first")
            return
        
        selected_device = st.selectbox(
            "Choose device",
            device_choices,
            format_func=lambda x: x[1]
        )
        
        if selected_device:
            device_id = selected_device[0]
            device_config = get_device(device_id)
            
            if device_config:
                st.success(f"✅ Connected to: {selected_device[1]}")
                st.info(f"📍 {device_config.get('host', 'Unknown')}")
            else:
                st.error("❌ Device configuration error")
                return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="feature-card"><h3>💭 Natural Language Input</h3></div>', unsafe_allow_html=True)
        
        if mode == "🚀 Push Configuration":
            nl_intent = st.text_area(
                "Describe what you want to configure",
                placeholder="Example: Enable OSPF on interface Gi0/0 and set the cost to 10",
                height=120
            )
        elif mode == "📊 Retrieve Information":
            nl_intent = st.text_area(
                "Ask a network question",
                placeholder="Example: Show me the OSPF neighbors on this router",
                height=120
            )
        else:  # Validate Configuration
            nl_intent = st.text_area(
                "Describe what configuration to validate",
                placeholder="Example: Validate the OSPF configuration and show neighbors",
                height=120
            )
        
        # Generate button
        if st.button("🤖 Generate Commands", use_container_width=True) and nl_intent:
            with st.spinner("🧠 AI is processing your request..."):
                # Step 1: Get context from ChromaDB
                context = chroma_search(nl_intent)
                
                # Step 2: Generate raw commands
                system_prompt = (
                    "You are a Cisco IOS expert. Translate the user's intention into CLI "
                    + ("configuration commands" if mode == "🚀 Push Configuration" else "show/exec commands for read-only retrieval")
                    + ". Provide the plain command lines only, no markdown, no explanations."
                )
                
                if context:
                    system_prompt += f"\n\nRelevant Cisco documentation:\n{context}"
                
                raw_output = run_ollama(
                    prompt=nl_intent,
                    system=system_prompt,
                    temperature=0.1
                )
                
                # Step 3: Clean and validate with CrewAI
                if mode == "🚀 Push Configuration":
                    cleaned_output = clean_with_agent(raw_output)
                else:
                    cleaned_output = raw_output
                
                commands = split_commands(cleaned_output)
                
                # Store in session state
                st.session_state.generated_commands = commands
                st.session_state.raw_output = raw_output
                st.session_state.cleaned_output = cleaned_output
                st.session_state.nl_intent = nl_intent
                st.session_state.mode = mode
                st.session_state.device_id = device_id
    
    with col2:
        st.markdown('<div class="feature-card"><h3>📊 Status</h3></div>', unsafe_allow_html=True)
        
        # System status
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                st.success("🟢 Ollama: Running")
            else:
                st.error("🔴 Ollama: Error")
        except:
            st.error("🔴 Ollama: Not reachable")
        
        # ChromaDB status
        try:
            cc = ChromaClient()
            cc.heartbeat()
            st.success("🟢 ChromaDB: Connected")
        except:
            st.warning("🟡 ChromaDB: Not available")
        
        # Database status
        try:
            with sqlite3.connect(DEVICES_DB) as conn:
                conn.execute("SELECT 1")
            st.success("🟢 Database: Connected")
        except:
            st.error("🔴 Database: Error")
    
    # Display generated commands
    if hasattr(st.session_state, 'generated_commands') and st.session_state.generated_commands:
        st.markdown('<div class="feature-card"><h3>🔧 Generated Commands</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.code("\n".join(st.session_state.generated_commands), language="bash")
        
        with col2:
            # Command validation
            valid_commands, invalid_commands = validate_cisco_commands(st.session_state.generated_commands)
            
            if valid_commands:
                st.success(f"✅ {len(valid_commands)} valid commands")
            if invalid_commands:
                st.warning(f"⚠️ {len(invalid_commands)} potentially invalid commands")
        
        # Action buttons
        if st.session_state.mode == "🚀 Push Configuration":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Deploy Configuration", use_container_width=True, type="primary"):
                    deploy_configuration(st.session_state.device_id, st.session_state.generated_commands)
            
            with col2:
                if st.button("🔍 Validate Commands", use_container_width=True):
                    validate_commands_detailed(st.session_state.generated_commands)
            
            with col3:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.rerun()
        
        elif st.session_state.mode == "📊 Retrieve Information":
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Execute Commands", use_container_width=True, type="primary"):
                    execute_retrieval_commands(st.session_state.device_id, st.session_state.generated_commands)
            
            with col2:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.rerun()
        
        else:  # Validate Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Generate Validation", use_container_width=True, type="primary"):
                    generate_and_execute_validation(st.session_state.device_id, st.session_state.nl_intent, st.session_state.generated_commands)
            
            with col2:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.rerun()

def deploy_configuration(device_id: int, commands: List[str]):
    """Deploy configuration to device."""
    st.markdown('<div class="feature-card"><h3>🚀 Configuration Deployment</h3></div>', unsafe_allow_html=True)
    
    with st.spinner("🔌 Connecting to device..."):
        try:
            device_config = get_device(device_id)
            output = netmiko_send_config(device_config, commands)
            
            st.success("✅ Configuration deployed successfully!")
            st.markdown('<div class="success-message"><h4>Deployment Output:</h4></div>', unsafe_allow_html=True)
            st.code(output, language="bash")
            
            # Generate validation commands
            st.markdown('<div class="feature-card"><h3>🔍 Post-Deployment Validation</h3></div>', unsafe_allow_html=True)
            
            validation_commands = generate_validation_commands(
                st.session_state.nl_intent,
                commands
            )
            
            if validation_commands:
                st.info("🔍 Generated validation commands:")
                st.code("\n".join(validation_commands), language="bash")
                
                if st.button("🔍 Execute Validation", use_container_width=True):
                    with st.spinner("🔍 Running validation..."):
                        try:
                            validation_output = netmiko_send_commands(device_config, validation_commands)
                            st.markdown('<div class="feature-card"><h3>📊 Validation Results</h3></div>', unsafe_allow_html=True)
                            st.code(validation_output, language="bash")
                        except Exception as e:
                            st.error(f"❌ Validation failed: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Deployment failed: {str(e)}")

def execute_retrieval_commands(device_id: int, commands: List[str]):
    """Execute retrieval commands on device."""
    st.markdown('<div class="feature-card"><h3>📊 Information Retrieval</h3></div>', unsafe_allow_html=True)
    
    with st.spinner("🔌 Executing commands..."):
        try:
            device_config = get_device(device_id)
            output = netmiko_send_commands(device_config, commands)
            
            st.success("✅ Commands executed successfully!")
            st.code(output, language="bash")
            
        except Exception as e:
            st.error(f"❌ Execution failed: {str(e)}")

def validate_commands_detailed(commands: List[str]):
    """Detailed command validation."""
    st.markdown('<div class="feature-card"><h3>🔍 Command Validation</h3></div>', unsafe_allow_html=True)
    
    valid_commands, invalid_commands = validate_cisco_commands(commands)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"✅ Valid Commands ({len(valid_commands)})")
        if valid_commands:
            for cmd in valid_commands:
                st.code(cmd, language="bash")
    
    with col2:
        if invalid_commands:
            st.warning(f"⚠️ Potentially Invalid Commands ({len(invalid_commands)})")
            for cmd in invalid_commands:
                st.code(cmd, language="bash")
        else:
            st.success("✅ All commands appear valid!")

def generate_and_execute_validation(device_id: int, intent: str, commands: List[str]):
    """Generate and execute validation commands."""
    st.markdown('<div class="feature-card"><h3>🔍 Configuration Validation</h3></div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Generating validation commands..."):
        validation_commands = generate_validation_commands(intent, commands)
        
        if validation_commands:
            st.info("🔍 Generated validation commands:")
            st.code("\n".join(validation_commands), language="bash")
            
            if st.button("🔍 Execute Validation", use_container_width=True):
                with st.spinner("🔍 Running validation..."):
                    try:
                        device_config = get_device(device_id)
                        validation_output = netmiko_send_commands(device_config, validation_commands)
                        
                        st.success("✅ Validation completed!")
                        st.code(validation_output, language="bash")
                        
                    except Exception as e:
                        st.error(f"❌ Validation failed: {str(e)}")
        else:
            st.warning("⚠️ Could not generate validation commands")

# Initialize database if needed
def init_database():
    """Initialize the devices database with sample data."""
    try:
        with sqlite3.connect(DEVICES_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL,
                    device_type TEXT NOT NULL
                )
            """)
            
            # Check if we have any devices
            cursor = conn.execute("SELECT COUNT(*) FROM devices")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Add sample devices
                sample_devices = [
                    ("R15", "172.16.39.102", "admin", "cisco", "cisco_ios_telnet"),
                    ("R16", "172.16.39.103", "admin", "cisco", "cisco_ios_telnet"),
                    ("R17", "172.16.39.104", "admin", "cisco", "cisco_ios_telnet"),
                ]
                
                for device in sample_devices:
                    conn.execute(
                        "INSERT INTO devices (name, host, username, password, device_type) VALUES (?, ?, ?, ?, ?)",
                        device
                    )
                
                conn.commit()
                st.success("✅ Database initialized with sample devices")
            
    except Exception as e:
        st.error(f"❌ Database initialization failed: {str(e)}")

# Call initialization
init_database() 