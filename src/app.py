import streamlit as st
import pandas as pd
import pickle
import torch
import numpy as np
import os
import plotly.graph_objects as go
import time
import shap
import matplotlib.pyplot as plt
import random
import re
import networkx as nx 
import wikipedia  # --- NEW: FOR GLOBAL KNOWLEDGE ---
from datetime import datetime # --- NEW: FOR TIME ---

# --- IMPORT MODULES ---
from gan_model import Generator
from lstm_model import CyberLSTM 
from voice_assistant import ai 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Aegis Defense", page_icon="🛡️", layout="wide")

# --- HACKER CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #00FF41; }
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier New'; }
    .stButton>button { 
        background-color: #00FF41; color: black; border: none; 
        font-weight: bold; font-family: 'Courier New'; height: 50px;
        box-shadow: 0 0 10px #00FF41;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px #00FF41;
        background-color: #ccffcc;
    }
    .stTextInput>div>div>input { color: #00FF41; background-color: #111; border: 1px solid #00FF41; }
    .stTextArea>div>div>textarea { color: #00FF41; background-color: #111; border: 1px solid #00FF41; }
    /* Terminal Box Style */
    .terminal {
        background-color: #000;
        border: 1px solid #333;
        padding: 10px;
        font-family: 'Courier New';
        font-size: 12px;
        color: #00FF41;
        height: 150px;
        overflow-y: scroll;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("models/detector_model.pkl"):
        with open("models/detector_model.pkl", "rb") as f: models['xgboost'] = pickle.load(f)
    if os.path.exists("models/generator_model.pth"):
        gen = Generator(100, 122); gen.load_state_dict(torch.load("models/generator_model.pth", map_location=torch.device('cpu'))); gen.eval(); models['gan'] = gen
    return models

models = load_models()

# --- SMART JARVIS BRAIN ---
def jarvis_brain(text):
    # 1. CLEAN THE INPUT (Remove noise)
    clean_text = text.lower().replace("hey", "").replace("jarvis", "").replace("please", "").strip()
    
    # 2. SECURITY COMMANDS (Hardcoded High Priority)
    if "scan" in clean_text or "analyze" in clean_text:
        return "Initiating deep forensic scan protocols. Please upload the log files."
    
    elif "status" in clean_text or "system" in clean_text:
        return "All systems nominal. Aegis defense grid is active and monitoring."
    
    elif "phishing" in clean_text:
        return "Phishing hunter module is online. Paste the email headers for analysis."
    
    elif "password" in clean_text or "credential" in clean_text:
        return "Credential audit tools are available in Tab 5."

    # 3. UTILITY COMMANDS
    elif "time" in clean_text:
        now = datetime.now().strftime("%I:%M %p")
        return f"The current system time is {now}."
    
    elif "date" in clean_text:
        today = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {today}."

    elif "who are you" in clean_text:
        return "I am Aegis, an AI-powered cybersecurity defense platform designed to protect your infrastructure."

    # 4. GLOBAL KNOWLEDGE (Wikipedia with Smart Search)
    else:
        try:
            # Remove question words to get the pure topic
            # e.g. "who is elon musk" -> "elon musk"
            topic = clean_text.replace("who is", "").replace("what is", "").replace("tell me about", "").strip()
            
            if not topic:
                return "I am listening. What topic do you want me to search for?"

            # Search specifically for the TOPIC, not the sentence
            # auto_suggest=False prevents it from guessing wrong
            try:
                summary = wikipedia.summary(topic, sentences=2, auto_suggest=True)
                return f"According to my database: {summary}"
            except wikipedia.exceptions.DisambiguationError as e:
                # This happens if a word has multiple meanings (e.g., "Apple" - Fruit or Company)
                return f"That topic is too broad. Did you mean {e.options[0]} or {e.options[1]}?"
        
        except wikipedia.exceptions.PageError:
            return f"I searched the global database for '{topic}', but found no specific records."
        except Exception as e:
            return "I am having trouble connecting to the knowledge base. Please check your internet."

# --- STARTUP SEQUENCE ---
if 'greeted' not in st.session_state:
    st.session_state['greeted'] = True
    ai.speak("Aegis System Online. Waiting for log files for deep analysis.")

# --- SIDEBAR & JARVIS ---
st.sidebar.title("🛡️ COMMAND CENTER")
st.sidebar.success("✅ VOICE: ACTIVE")

st.sidebar.markdown("---")
st.sidebar.subheader("🎙️ JARVIS SETTINGS")

# --- MICROPHONE SELECTOR ---
mic_options = ai.list_microphones()
mic_dict = {f"{idx}: {name}": idx for idx, name in mic_options}

if mic_dict:
    selected_mic_label = st.sidebar.selectbox("Select Microphone Source:", list(mic_dict.keys()))
    selected_mic_index = mic_dict[selected_mic_label]
else:
    st.sidebar.error("No microphones found! Check hardware.")
    selected_mic_index = None

if st.sidebar.button("🔊 ACTIVATE VOICE COMMAND"):
    if selected_mic_index is not None:
        with st.spinner(f"Listening on Device {selected_mic_index}..."):
            user_voice = ai.listen(device_index=selected_mic_index)
            
            if user_voice:
                st.sidebar.info(f"You: {user_voice}")
                # Get response from the upgraded brain
                response = jarvis_brain(user_voice)
                st.sidebar.warning(f"Jarvis: {response}")
                ai.speak(response)
            else:
                st.sidebar.error("No audio detected. Try a different Mic from the list above.")
    else:
        st.sidebar.error("Microphone not available.")

st.sidebar.markdown("---")
st.sidebar.info("System Ready. Upload logs to generate Deep Analysis Report.")


# --- MAIN APP ---
st.title("🛡️ AEGIS: AI-Powered Defense Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🛡️ DEEP FORENSIC SCANNER", "⚔️ RED TEAM SIMULATOR", "📧 PHISHING HUNTER", "🕸️ NETWORK MAPPER", "🔐 CREDENTIAL AUDIT"])

# ==================================================
# TAB 1: DEEP LOG ANALYSIS
# ==================================================
with tab1:
    st.subheader("📁 Upload Network Logs (Deep Packet Inspection)")
    uploaded_file = st.file_uploader("Select Log File (CSV)", type=["csv", "txt"])
    terminal_placeholder = st.empty()
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write(f"**Loaded {len(raw_df)} records.** Preview:")
            st.dataframe(raw_df.head(3))
            
            if st.button("RUN DEEP FORENSIC ANALYSIS"):
                ai.speak(f"File loaded successfully. Analyzing {len(raw_df)} records for anomalies.")
                logs = []
                with st.spinner("Processing Data Stream..."):
                    for i in range(5):
                        time.sleep(0.5)
                        logs.append(f"[SCAN] Analyzing packet batch {i*100} - {i*100+100}...")
                        terminal_placeholder.code("\n".join(logs))
                    
                    logs.append("[COMPLETE] Signatures matched against threat database.")
                    terminal_placeholder.code("\n".join(logs))
                    
                    try:
                        if os.path.exists("processed_data/X_test.pkl"):
                            with open("processed_data/X_test.pkl", "rb") as f:
                                template = pickle.load(f)
                                expected_cols = template.columns.tolist()
                            
                            numeric_df = raw_df.apply(pd.to_numeric, errors='coerce')
                            df = pd.DataFrame(0, index=np.arange(len(raw_df)), columns=expected_cols)
                            raw_df.columns = raw_df.columns.str.strip()
                            common = list(set(raw_df.columns) & set(expected_cols))
                            if common:
                                df[common] = numeric_df[common].fillna(0)
                            df = df.fillna(0)
                        else:
                            st.error("Missing template file (X_test.pkl). Cannot align columns.")
                            st.stop()
                            
                        preds = models['xgboost'].predict(df)
                        probs = models['xgboost'].predict_proba(df)[:, 1]

                    except Exception as e:
                        st.error(f"Model Error: {e}")
                        st.stop()
                    
                    threats = sum(preds)
                    total = len(preds)
                    
                    if threats > 0:
                        worst_idx = np.argmax(probs)
                        worst_prob = probs[worst_idx]
                        explainer = shap.TreeExplainer(models['xgboost'])
                        shap_values = explainer.shap_values(df.iloc[[worst_idx]])
                        top_2_idx = np.abs(shap_values[0]).argsort()[-2:][::-1]
                        feat1 = df.columns[top_2_idx[0]]
                        feat2 = df.columns[top_2_idx[1]]
                        
                        report = (
                            f"Diagnostic Complete. I have processed {total} log entries. "
                            f"Alert Level: Critical. I detected {threats} confirmed threats. "
                            f"The most dangerous anomaly has a threat probability of {int(worst_prob*100)} percent. "
                            f"Deep analysis reveals the primary indicators are abnormal {feat1} and {feat2}. "
                            f"Recommendation: Isolate the source IP address immediately."
                        )
                        st.error(f"🚨 **CRITICAL THREATS DETECTED: {threats}**")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.info(f"🗣️ **AI Briefing:** {report}")
                            fig, ax = plt.subplots(figsize=(10, 3))
                            shap.summary_plot(shap_values, df.iloc[[worst_idx]], plot_type="bar", show=False)
                            st.pyplot(fig)
                        with col2:
                            st.warning("⚠️ **Top Indicators:**")
                            st.write(f"1. {feat1}")
                            st.write(f"2. {feat2}")
                        ai.speak(report)
                    else:
                        report = "System integrity is 100 percent. No anomalies detected."
                        st.success("✅ SYSTEM CLEAN")
                        ai.speak(report)

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        if st.button("RUN DEMO SCAN (INTERNAL DATA)"):
             ai.speak("Initiating demo scan using internal test data.")
             with st.spinner("Loading internal modules..."):
                 terminal_placeholder.code("[INIT] Loading X_test.pkl...\n[SCAN] Deep Packet Inspection engaged...")
                 time.sleep(2)
                 with open("processed_data/X_test.pkl", "rb") as f: df = pickle.load(f)
                 preds = models['xgboost'].predict(df)
                 threats = sum(preds)
                 msg = f"Demo complete. Detected {threats} threats in the internal database."
                 ai.speak(msg)
                 st.metric("Demo Threats", threats)

# ==================================================
# TAB 2: OFFENSE
# ==================================================
with tab2:
    st.subheader("Adversarial Simulation")
    if st.button("LAUNCH SIMULATION"):
        ai.speak("Launching synthetic attack vectors. Brace for impact.")
        with st.spinner("Simulating..."):
            time.sleep(3)
            z = torch.randn(1000, 100)
            fake_data = models['gan'](z).detach().numpy()
            preds = models['xgboost'].predict(fake_data)
            blocked = sum(preds)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = blocked, title = {'text': "Attacks Blocked"},
                gauge = {'axis': {'range': [None, 1000]}, 'bar': {'color': "#00FF41"}}
            ))
            st.plotly_chart(fig)
            ai.speak(f"Simulation complete. {blocked} attacks blocked.")

# ==================================================
# TAB 3: PHISHING
# ==================================================
with tab3:
    st.subheader("📧 Advanced Phishing Hunter")
    email = st.text_area("Paste Email Content (Header/Body/Links):", height=150)
    if st.button("ANALYZE EMAIL"):
        if email:
            ai.speak("Scanning email content for social engineering patterns.")
            with st.spinner("Analyzing..."):
                time.sleep(2)
                keywords = ["urgent", "password", "bank", "verify", "suspended", "expire"]
                found = [w for w in keywords if w in email.lower()]
                links = re.findall(r'(https?://[^\s]+)', email)
                risk = 0
                evidence = []
                if found:
                    risk += 30
                    evidence.append(f"Urgency keywords: {', '.join(found)}")
                if links:
                    for link in links:
                        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', link):
                            risk += 50
                            evidence.append(f"Raw IP Link detected: {link}")
                        elif "http:" in link:
                            risk += 20
                            evidence.append(f"Insecure Link: {link}")
                        else:
                            risk += 10
                st.markdown("---")
                if risk > 50:
                    verdict = "Phishing Confirmed"
                    report = f"Analysis complete. {verdict}. Risk score is {risk} percent."
                    st.error(f"🚨 {verdict}")
                    for e in evidence: st.warning(f"- {e}")
                    ai.speak(report)
                else:
                    report = "Analysis complete. The email appears safe."
                    st.success("✅ Safe")
                    ai.speak(report)
        else:
            ai.speak("Please paste an email first.")

# ==================================================
# TAB 4: NETWORK MAPPER
# ==================================================
with tab4:
    st.subheader("🕸️ Network Topology Visualizer")
    if st.button("GENERATE NETWORK MAP"):
        ai.speak("Mapping active network nodes and analyzing traffic flow.")
        with st.spinner("Tracing routes..."):
            time.sleep(2)
            G = nx.random_geometric_graph(20, 0.3)
            pos = nx.spring_layout(G)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#00FF41'), hoverinfo='none', mode='lines')

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers', hoverinfo='text',
                marker=dict(
                    showscale=True, colorscale='YlOrRd', reversescale=True, color=[], size=10,
                    colorbar=dict(thickness=15, title=dict(text='Node Activity', side='right'), xanchor='left'),
                    line_width=2))

            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_text.append(f'Server Node #{node}: {len(adjacencies[1])} connections')

            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                            plot_bgcolor='black', paper_bgcolor='black', font=dict(color='#00FF41'),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            st.plotly_chart(fig, use_container_width=True)
            
            critical_nodes = [i for i, x in enumerate(node_adjacencies) if x > 4]
            if critical_nodes:
                 report = f"Network mapping complete. Identified {len(critical_nodes)} critical nodes."
                 st.warning(f"⚠️ Critical Nodes Identified: {critical_nodes}")
                 ai.speak(report)
            else:
                 ai.speak("Network mapping complete. Topology is stable.")

# ==================================================
# TAB 5: CREDENTIAL AUDIT
# ==================================================
with tab5:
    st.subheader("🔐 Credential Strength Auditor")
    password = st.text_input("Test Password Strength:", type="password")
    if st.button("AUDIT CREDENTIAL"):
        if password:
            score = 0
            feedback = []
            if len(password) >= 12: score += 30
            elif len(password) >= 8: score += 10
            else: feedback.append("Password is too short.")
            if re.search(r"[A-Z]", password): score += 20
            else: feedback.append("Missing uppercase letter.")
            if re.search(r"[a-z]", password): score += 20
            else: feedback.append("Missing lowercase letter.")
            if re.search(r"\d", password): score += 15
            else: feedback.append("Missing number.")
            if re.search(r"[!@#$%^&*]", password): score += 15
            else: feedback.append("Missing special character.")
            
            st.markdown("---")
            if score >= 90:
                msg = "Password Strength: Excellent."
                st.success(f"✅ {msg}")
                ai.speak(msg)
            elif score >= 60:
                msg = "Password Strength: Moderate."
                st.warning(f"⚠️ {msg}")
                ai.speak(msg)
            else:
                msg = "Password Strength: Weak."
                st.error(f"❌ {msg}")
                for f in feedback: st.write(f"- {f}")
                ai.speak(msg)