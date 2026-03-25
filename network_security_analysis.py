# ============================================================
#  NETWORK SECURITY THREAT ANALYSIS
#  Author: Shrey Dukare | BSIOTR, Pune | B.E. Computer Engineering
#  Tools: Python, Pandas, Matplotlib, Seaborn
#  Based on: Computer Security coursework — CIA Triad Framework
#  Copy-paste this entire file into ONE Google Colab cell
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print("✅ Libraries imported!")

# ─────────────────────────────────────────────────────────────
# STEP 1: Build the Attack Taxonomy Dataset
# ─────────────────────────────────────────────────────────────
attacks = pd.DataFrame([
    # Name, Category, CIA Impact, Detection Difficulty, Severity, Description
    ("Eavesdropping",         "Passive", "Confidentiality", "Hard",   "High",   "Attacker silently intercepts network traffic without altering it."),
    ("Traffic Analysis",      "Passive", "Confidentiality", "Hard",   "Medium", "Analyzing packet patterns to infer communication even if encrypted."),
    ("Port Scanning",         "Passive", "Confidentiality", "Medium", "Low",    "Probing open ports on a target system to map its services."),
    ("Packet Sniffing",       "Passive", "Confidentiality", "Hard",   "High",   "Capturing data packets on a network using tools like Wireshark."),
    ("OS Fingerprinting",     "Passive", "Confidentiality", "Medium", "Low",    "Identifying the target OS through response patterns to crafted packets."),
    ("DoS Attack",            "Active",  "Availability",    "Medium", "High",   "Flooding a target with traffic to exhaust its resources."),
    ("DDoS Attack",           "Active",  "Availability",    "Hard",   "Critical","Coordinated DoS using multiple compromised machines (botnet)."),
    ("Replay Attack",         "Active",  "Integrity",       "Hard",   "High",   "Re-transmitting captured valid data packets to deceive the receiver."),
    ("Man-in-the-Middle",     "Active",  "Confidentiality", "Hard",   "Critical","Attacker intercepts and possibly alters communication between two parties."),
    ("IP Spoofing",           "Active",  "Integrity",       "Medium", "High",   "Forging the source IP address in packets to impersonate another host."),
    ("ARP Poisoning",         "Active",  "Integrity",       "Medium", "High",   "Sending fake ARP replies to link attacker's MAC with a legitimate IP."),
    ("DNS Spoofing",          "Active",  "Integrity",       "Hard",   "Critical","Corrupting DNS cache to redirect users to malicious websites."),
    ("Phishing",              "Active",  "Confidentiality", "Easy",   "High",   "Tricking users into revealing credentials via fake websites or emails."),
    ("SQL Injection",         "Active",  "Integrity",       "Easy",   "Critical","Injecting malicious SQL queries to manipulate or dump a database."),
    ("Buffer Overflow",       "Active",  "Availability",    "Hard",   "Critical","Overwriting memory beyond a buffer to execute arbitrary code."),
    ("Session Hijacking",     "Active",  "Confidentiality", "Medium", "High",   "Stealing or forging a valid session token to impersonate a user."),
    ("Cross-Site Scripting",  "Active",  "Integrity",       "Easy",   "Medium", "Injecting malicious scripts into web pages viewed by other users."),
    ("Ransomware",            "Active",  "Availability",    "Easy",   "Critical","Encrypting victim's files and demanding ransom for the decryption key."),
    ("Rootkit",               "Active",  "Integrity",       "Hard",   "Critical","Malware that hides itself and provides persistent privileged access."),
    ("Keylogger",             "Active",  "Confidentiality", "Hard",   "High",   "Records keystrokes to capture passwords and sensitive information."),
])

attacks.columns = ['Attack', 'Category', 'CIA_Target', 'Detection', 'Severity', 'Description']
print(f"✅ Attack taxonomy built: {len(attacks)} attack types")
print(attacks[['Attack', 'Category', 'CIA_Target', 'Severity']].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 2: CIA Triad Overview
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("THE CIA TRIAD — CORE FRAMEWORK IN NETWORK SECURITY")
print("="*60)
cia_info = {
    "Confidentiality": "Ensures information is accessible only to authorized parties.\nThreats: Eavesdropping, Phishing, Session Hijacking, Packet Sniffing",
    "Integrity":       "Ensures data is accurate and has not been tampered with.\nThreats: Replay Attack, ARP Poisoning, DNS Spoofing, SQL Injection",
    "Availability":    "Ensures systems and data are accessible when needed.\nThreats: DoS, DDoS, Ransomware, Buffer Overflow"
}
for pillar, desc in cia_info.items():
    print(f"\n🔷 {pillar}:\n   {desc}")

# ─────────────────────────────────────────────────────────────
# STEP 3: EDA Visualizations
# ─────────────────────────────────────────────────────────────
severity_order = ['Low', 'Medium', 'High', 'Critical']
severity_colors = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e67e22', 'Critical': '#e74c3c'}
attacks['Severity'] = pd.Categorical(attacks['Severity'], categories=severity_order, ordered=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Network Security Threat Analysis — Attack Taxonomy', fontsize=15, fontweight='bold')

# Plot 1: Active vs Passive
cat_counts = attacks['Category'].value_counts()
axes[0,0].pie(cat_counts, labels=cat_counts.index, autopct='%1.0f%%',
              colors=['#e74c3c', '#3498db'], startangle=90, explode=[0.05, 0])
axes[0,0].set_title('Active vs Passive Attacks', fontweight='bold')

# Plot 2: CIA Triad Target Distribution
cia_counts = attacks['CIA_Target'].value_counts()
bars = axes[0,1].bar(cia_counts.index, cia_counts.values,
                     color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', width=0.5)
axes[0,1].set_title('Attacks by CIA Triad Target', fontweight='bold')
axes[0,1].set_ylabel('Number of Attack Types')
for bar, v in zip(bars, cia_counts.values):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=12)

# Plot 3: Severity Distribution
sev_counts = attacks['Severity'].value_counts().reindex(severity_order)
bar_colors = [severity_colors[s] for s in severity_order]
bars2 = axes[1,0].bar(severity_order, sev_counts.values, color=bar_colors, edgecolor='black', width=0.5)
axes[1,0].set_title('Attack Severity Distribution', fontweight='bold')
axes[1,0].set_ylabel('Number of Attacks')
for bar, v in zip(bars2, sev_counts.values):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v), ha='center', fontweight='bold', fontsize=12)

# Plot 4: Heatmap — Category vs CIA Target
heat_data = pd.crosstab(attacks['CIA_Target'], attacks['Category'])
sns.heatmap(heat_data, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=1, ax=axes[1,1], cbar=False, annot_kws={'size': 14})
axes[1,1].set_title('CIA Target vs Attack Category (Heatmap)', fontweight='bold')
axes[1,1].set_xlabel('Attack Category')
axes[1,1].set_ylabel('CIA Triad Target')

plt.tight_layout()
plt.savefig('attack_taxonomy_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# STEP 4: Detection Difficulty vs Severity Matrix
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

detect_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
sev_map    = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
cat_color  = {'Passive': '#3498db', 'Active': '#e74c3c'}

attacks['det_num'] = attacks['Detection'].map(detect_map)
attacks['sev_num'] = attacks['Severity'].map(sev_map)

for _, row in attacks.iterrows():
    ax.scatter(row['det_num'], row['sev_num'],
               color=cat_color[row['Category']], s=200, zorder=3,
               edgecolors='black', linewidth=0.8)
    ax.annotate(row['Attack'], (row['det_num'], row['sev_num']),
                textcoords='offset points', xytext=(6, 4), fontsize=7.5)

ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Easy', 'Medium', 'Hard'], fontsize=11)
ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(['Low', 'Medium', 'High', 'Critical'], fontsize=11)
ax.set_xlabel('Detection Difficulty →', fontsize=12)
ax.set_ylabel('Severity →', fontsize=12)
ax.set_title('Risk Matrix — Detection Difficulty vs Severity', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

legend_handles = [mpatches.Patch(color=c, label=l) for l, c in cat_color.items()]
ax.legend(handles=legend_handles, title='Category', fontsize=10)

# Highlight danger zone
ax.axhspan(3.5, 4.5, alpha=0.08, color='red')
ax.text(3.05, 3.75, '⚠ DANGER ZONE', color='red', fontsize=9, fontstyle='italic')

plt.tight_layout()
plt.savefig('risk_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# STEP 5: Passive vs Active — Detailed Comparison
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASSIVE vs ACTIVE ATTACKS — DETAILED COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Property'         : ['Modifies Data?', 'Detectable?', 'Goal', 'Example', 'CIA Impact', 'Prevention'],
    'Passive Attack'   : ['No', 'Very Hard', 'Collect information', 'Eavesdropping, Sniffing', 'Confidentiality', 'Encryption'],
    'Active Attack'    : ['Yes', 'Easier (leaves traces)', 'Disrupt / Damage', 'DoS, MitM, SQL Injection', 'All three (C, I, A)', 'Firewalls, IDS/IPS'],
})
print(comparison.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 6: Mitigation Strategies
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MITIGATION STRATEGIES BY CIA PILLAR")
print("="*60)

mitigations = {
    "Confidentiality": ["TLS/SSL encryption for data in transit", "VPN for secure remote access",
                        "Strong authentication (MFA)", "Principle of least privilege"],
    "Integrity":       ["Digital signatures & checksums (MD5, SHA-256)", "Certificate-based authentication",
                        "Input validation to prevent SQL injection/XSS", "Audit logs & change monitoring"],
    "Availability":    ["DDoS protection & rate limiting", "Redundancy & failover systems",
                        "Regular backups (3-2-1 rule)", "Intrusion Detection Systems (IDS/IPS)"],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Mitigation Strategies by CIA Triad Pillar', fontsize=13, fontweight='bold')
cia_colors = {'Confidentiality': '#3498db', 'Integrity': '#e74c3c', 'Availability': '#2ecc71'}

for ax, (pillar, items) in zip(axes, mitigations.items()):
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    ax.set_title(f'🔷 {pillar}', fontweight='bold', color=cia_colors[pillar], fontsize=12, pad=10)
    y = 0.85
    for item in items:
        ax.text(0.05, y, f'✅  {item}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', wrap=True)
        y -= 0.22
    rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                          edgecolor=cia_colors[pillar], linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig('mitigations.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# STEP 7: Full Taxonomy Table
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FULL ATTACK TAXONOMY TABLE")
print("="*60)
display_cols = ['Attack', 'Category', 'CIA_Target', 'Severity', 'Detection', 'Description']
print(attacks[display_cols].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 8: Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"""
Total attacks analyzed : {len(attacks)}
Passive attacks        : {len(attacks[attacks['Category']=='Passive'])}
Active attacks         : {len(attacks[attacks['Category']=='Active'])}
Critical severity      : {len(attacks[attacks['Severity']=='Critical'])}
Hard to detect         : {len(attacks[attacks['Detection']=='Hard'])}

Most targeted CIA pillar : {attacks['CIA_Target'].value_counts().index[0]}
Most common severity     : {attacks['Severity'].value_counts().index[0]}

Key Findings:
  • {len(attacks[attacks['Category']=='Active'])/len(attacks)*100:.0f}% of attacks are Active (modify data or disrupt services)
  • Critical + Hard-to-detect attacks = most dangerous combination
  • Confidentiality is the most targeted CIA pillar
  • Encryption + IDS/IPS are the most universally effective countermeasures

Framework used: CIA Triad (Confidentiality, Integrity, Availability)

Author : Shrey Dukare | BSIOTR, Pune
Email  : shreydokre@gmail.com
""")
print("✅ Project Complete!")
