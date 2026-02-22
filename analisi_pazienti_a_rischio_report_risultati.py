import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAZIONE REPORT
# =============================================================================
report_file = f"report_analisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
html_content = []

def add_to_report(title, content, type='text'):
    """Aggiunge contenuto al report HTML"""
    if type == 'title1':
        html_content.append(f"<h1>{title}</h1>")
    elif type == 'title2':
        html_content.append(f"<h2>{title}</h2>")
    elif type == 'title3':
        html_content.append(f"<h3>{title}</h3>")
    elif type == 'code':
        html_content.append(f"<pre>{content}</pre>")
    elif type == 'df':
        html_content.append(f"<div class='table-container'>{content.to_html(classes='dataframe')}</div>")
    elif type == 'image':
        html_content.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{content}' loading='lazy'><div class='chart-title'>{title}</div></div>")
    else:
        html_content.append(f"<p>{content}</p>")

def fig_to_base64(fig):
    """Converte una figura matplotlib in stringa base64 per embedding HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# =============================================================================
# CONNESSIONE DATABASE E ESTRAZIONE DATI
# =============================================================================
conn = sqlite3.connect('database_sanitario.db')

query = '''
SELECT 
    p.id_paziente,
    p.nome,
    p.cognome,
    p.eta,
    p.sesso,
    p.fumatore,
    p.attivita_fisica,
    AVG(a.grassi_saturi_g) as media_grassi_saturi,
    AVG(a.zuccheri_g) as media_zuccheri,
    AVG(a.calorie_totali) as media_calorie,
    e.colesterolo_totale,
    e.HDL,
    e.LDL,
    e.glicemia
FROM pazienti p
LEFT JOIN alimentazione a ON p.id_paziente = a.id_paziente
LEFT JOIN esami_clinici e ON p.id_paziente = e.id_paziente
GROUP BY p.id_paziente
'''

df = pd.read_sql(query, conn)
conn.close()

add_to_report("Report Analisi Dati Sanitari", "", 'title1')
add_to_report(f"Dataset: {df.shape[0]} pazienti, {df.shape[1]} variabili", "", 'text')
add_to_report("Prime 5 righe del dataset", "", 'title3')
add_to_report("", df.head(), 'df')

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
le_sesso = LabelEncoder()
le_fumatore = LabelEncoder()
le_attivita = LabelEncoder()

df['sesso_enc'] = le_sesso.fit_transform(df['sesso'])
df['fumatore_enc'] = le_fumatore.fit_transform(df['fumatore'])
df['attivita_enc'] = le_attivita.fit_transform(df['attivita_fisica'])

df['rapporto_HDL'] = df['HDL'] / df['colesterolo_totale']
df['rapporto_LDL_HDL'] = df['LDL'] / df['HDL']
df['rischio_metabolico'] = (df['colesterolo_totale'] / 200 + df['glicemia'] / 100) / 2
df['rischio_alto'] = (df['colesterolo_totale'] > 240).astype(int)

# =============================================================================
# ANALISI ESPLORATIVA
# =============================================================================
add_to_report("Analisi Esplorativa", "", 'title2')

stats_desc = df[['eta', 'media_grassi_saturi', 'media_zuccheri',
                 'colesterolo_totale', 'HDL', 'LDL', 'glicemia']].describe()
add_to_report("Statistiche descrittive", "", 'title3')
add_to_report("", stats_desc, 'df')

numeric_cols = ['eta', 'media_grassi_saturi', 'media_zuccheri', 'media_calorie',
                'colesterolo_totale', 'HDL', 'LDL', 'glicemia', 'rapporto_HDL',
                'rapporto_LDL_HDL', 'rischio_metabolico']

corr_matrix = df[numeric_cols].corr()
add_to_report("Correlazioni con colesterolo totale", "", 'title3')
corr_series = corr_matrix['colesterolo_totale'].sort_values(ascending=False)
add_to_report("", corr_series.to_frame().T, 'df')

# =============================================================================
# GENERAZIONE GRAFICI
# =============================================================================
add_to_report("Visualizzazioni", "", 'title2')
add_to_report("Clicca su ogni grafico per ingrandirlo", "", 'text')

charts_html = []
charts_html.append("<div class='charts-grid'>")

# Grafico 1: Distribuzione colesterolo
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.hist(df['colesterolo_totale'], bins=15, edgecolor='black', alpha=0.7, color='#AEC6CF')
ax1.axvline(df['colesterolo_totale'].mean(), color='#FFB347', linestyle='--', linewidth=2,
            label=f'Media: {df["colesterolo_totale"].mean():.1f}')
ax1.axvline(200, color='#77DD77', linestyle='--', linewidth=2, label='Soglia 200')
ax1.axvline(240, color='#F49AC2', linestyle='--', linewidth=2, label='Soglia 240')
ax1.set_title('Distribuzione Colesterolo', fontsize=12, fontweight='bold')
ax1.set_xlabel('Colesterolo (mg/dL)', fontsize=10)
ax1.set_ylabel('Frequenza', fontsize=10)
ax1.legend(fontsize=8)
ax1.set_facecolor('#F5F5F5')
fig1.patch.set_facecolor('#FFFFFF')
plt.tight_layout()
img1 = fig_to_base64(fig1)
charts_html.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{img1}' loading='lazy'><div class='chart-title'>Distribuzione Colesterolo</div></div>")

# Grafico 2: Boxplot per fumatore
fig2, ax2 = plt.subplots(figsize=(8, 5))
df.boxplot(column='colesterolo_totale', by='fumatore', ax=ax2,
           boxprops=dict(color='#779ECB'), whiskerprops=dict(color='#779ECB'),
           capprops=dict(color='#779ECB'), medianprops=dict(color='#FFB347', linewidth=2))
ax2.set_title('Colesterolo per abitudine al fumo', fontsize=12, fontweight='bold')
ax2.set_xlabel('Fumatore', fontsize=10)
ax2.set_ylabel('Colesterolo Totale', fontsize=10)
ax2.set_facecolor('#F5F5F5')
fig2.patch.set_facecolor('#FFFFFF')
plt.tight_layout()
img2 = fig_to_base64(fig2)
charts_html.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{img2}' loading='lazy'><div class='chart-title'>Colesterolo per Fumo</div></div>")

# Grafico 3: Boxplot per attività fisica
fig3, ax3 = plt.subplots(figsize=(8, 5))
df.boxplot(column='colesterolo_totale', by='attivita_fisica', ax=ax3,
           boxprops=dict(color='#B19CD9'), whiskerprops=dict(color='#B19CD9'),
           capprops=dict(color='#B19CD9'), medianprops=dict(color='#FFB347', linewidth=2))
ax3.set_title('Colesterolo per attività fisica', fontsize=12, fontweight='bold')
ax3.set_xlabel('Attività Fisica', fontsize=10)
ax3.set_ylabel('Colesterolo Totale', fontsize=10)
ax3.set_facecolor('#F5F5F5')
fig3.patch.set_facecolor('#FFFFFF')
plt.tight_layout()
img3 = fig_to_base64(fig3)
charts_html.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{img3}' loading='lazy'><div class='chart-title'>Colesterolo per Attività</div></div>")

# Grafico 4: Scatter colesterolo vs grassi saturi
fig4, ax4 = plt.subplots(figsize=(8, 5))
scatter = ax4.scatter(df['media_grassi_saturi'], df['colesterolo_totale'],
                      c=df['eta'], cmap='viridis', s=80, alpha=0.7, edgecolors='white')
ax4.set_title('Colesterolo vs Grassi Saturi', fontsize=12, fontweight='bold')
ax4.set_xlabel('Media Grassi Saturi (g)', fontsize=10)
ax4.set_ylabel('Colesterolo Totale', fontsize=10)
plt.colorbar(scatter, ax=ax4, label='Età')
ax4.set_facecolor('#F5F5F5')
fig4.patch.set_facecolor('#FFFFFF')
plt.tight_layout()
img4 = fig_to_base64(fig4)
charts_html.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{img4}' loading='lazy'><div class='chart-title'>Colesterolo vs Grassi</div></div>")

# Grafico 5: Scatter colesterolo vs zuccheri
fig5, ax5 = plt.subplots(figsize=(8, 5))
scatter = ax5.scatter(df['media_zuccheri'], df['colesterolo_totale'],
                      c=df['eta'], cmap='plasma', s=80, alpha=0.7, edgecolors='white')
ax5.set_title('Colesterolo vs Zuccheri', fontsize=12, fontweight='bold')
ax5.set_xlabel('Media Zuccheri (g)', fontsize=10)
ax5.set_ylabel('Colesterolo Totale', fontsize=10)
plt.colorbar(scatter, ax=ax5, label='Età')
ax5.set_facecolor('#F5F5F5')
fig5.patch.set_facecolor('#FFFFFF')
plt.tight_layout()
img5 = fig_to_base64(fig5)
charts_html.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{img5}' loading='lazy'><div class='chart-title'>Colesterolo vs Zuccheri</div></div>")

# Grafico 6: Heatmap correlazioni
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
            ax=ax6, linewidths=0.5, linecolor='white', annot_kws={'size': 8})
ax6.set_title('Matrice di Correlazione', fontsize=12, fontweight='bold')
ax6.tick_params(axis='x', rotation=45, labelsize=8)
ax6.tick_params(axis='y', labelsize=8)
fig6.patch.set_facecolor('#FFFFFF')
plt.tight_layout()
img6 = fig_to_base64(fig6)
charts_html.append(f"<div class='chart-item' onclick='openModal(this)'><img src='{img6}' loading='lazy'><div class='chart-title'>Matrice Correlazione</div></div>")

charts_html.append("</div>")
html_content.append(''.join(charts_html))

# =============================================================================
# MODELLO DI REGRESSIONE
# =============================================================================
add_to_report("Modello di Regressione - Predizione Colesterolo Totale", "", 'title2')

features_reg = ['eta', 'sesso_enc', 'fumatore_enc', 'attivita_enc',
                'media_grassi_saturi', 'media_zuccheri', 'glicemia']

X_reg = df[features_reg]
y_reg = df['colesterolo_totale']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = rf_reg.predict(X_test_reg)

r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = np.mean(np.abs(y_test_reg - y_pred_reg))

add_to_report(f"R² Score: {r2:.3f}", "", 'text')
add_to_report(f"RMSE: {rmse:.2f} mg/dL", "", 'text')
add_to_report(f"MAE: {mae:.2f} mg/dL", "", 'text')

feature_importance = pd.DataFrame({
    'feature': features_reg,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

add_to_report("Importanza delle feature", "", 'title3')
add_to_report("", feature_importance, 'df')

# =============================================================================
# MODELLO DI CLASSIFICAZIONE
# =============================================================================
add_to_report("Modello di Classificazione - Rischio Alto", "", 'title2')

features_clf = ['eta', 'sesso_enc', 'fumatore_enc', 'attivita_enc',
                'media_grassi_saturi', 'media_zuccheri', 'glicemia']

X_clf = df[features_clf]
y_clf = df['rischio_alto']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)

y_pred_clf = rf_clf.predict(X_test_clf)

add_to_report("Classification Report", "", 'title3')
report_dict = classification_report(y_test_clf, y_pred_clf,
                                  target_names=['Rischio Basso', 'Rischio Alto'],
                                  output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
add_to_report("", report_df, 'df')

cm = confusion_matrix(y_test_clf, y_pred_clf)
add_to_report("Matrice di Confusione", "", 'title3')
cm_df = pd.DataFrame(cm, index=['Actual Basso', 'Actual Alto'],
                     columns=['Predetto Basso', 'Predetto Alto'])
add_to_report("", cm_df, 'df')

feature_importance_clf = pd.DataFrame({
    'feature': features_clf,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

add_to_report("Importanza delle feature", "", 'title3')
add_to_report("", feature_importance_clf, 'df')

# =============================================================================
# PREDIZIONI SU NUOVI PAZIENTI
# =============================================================================
add_to_report("Predizioni su Nuovi Pazienti", "", 'title2')

nuovi_pazienti = pd.DataFrame({
    'eta': [35, 55, 70],
    'sesso': ['M', 'F', 'M'],
    'fumatore': ['NO', 'SI', 'NO'],
    'attivita_fisica': ['Alta', 'Bassa', 'Sedentario'],
    'media_grassi_saturi': [15, 35, 25],
    'media_zuccheri': [50, 110, 80],
    'glicemia': [85, 115, 95]
})

nuovi_pazienti['sesso_enc'] = le_sesso.transform(nuovi_pazienti['sesso'])
nuovi_pazienti['fumatore_enc'] = le_fumatore.transform(nuovi_pazienti['fumatore'])

attivita_map = {'Bassa': 0, 'Moderata': 1, 'Alta': 2}
nuovi_pazienti['attivita_enc'] = nuovi_pazienti['attivita_fisica'].map(
    lambda x: attivita_map.get(x, 0)
)

X_nuovi = nuovi_pazienti[features_reg]
colesterolo_pred = rf_reg.predict(X_nuovi)
rischio_pred = rf_clf.predict(X_nuovi)
rischio_prob = rf_clf.predict_proba(X_nuovi)[:, 1]

risultati = nuovi_pazienti[['eta', 'sesso', 'fumatore', 'attivita_fisica']].copy()
risultati['colesterolo_predetto'] = colesterolo_pred.astype(int)
risultati['prob_rischio_alto'] = (rischio_prob * 100).round(1)
risultati['classe_rischio'] = ['ALTO' if r == 1 else 'BASSO' for r in rischio_pred]

add_to_report("", risultati, 'df')

# =============================================================================
# REPORT FINALE
# =============================================================================
add_to_report("Report Finale", "", 'title2')

add_to_report(f"""
ANALISI COMPLETATA
------------------
Dataset: {df.shape[0]} pazienti analizzati

MODELLO REGRESSIONE
- R²: {r2:.3f}
- RMSE: {rmse:.2f} mg/dL
- Feature più importanti: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.3f})

MODELLO CLASSIFICAZIONE
- Accuracy: {rf_clf.score(X_test_clf, y_test_clf):.3f}
- Feature più importanti: {feature_importance_clf.iloc[0]['feature']} ({feature_importance_clf.iloc[0]['importance']:.3f})
""", "", 'text')

alto_rischio = df[df['rischio_alto'] == 1][['nome', 'cognome', 'eta', 'colesterolo_totale']]
add_to_report("Pazienti ad alto rischio nel dataset", "", 'title3')
if len(alto_rischio) > 0:
    add_to_report("", alto_rischio, 'df')
else:
    add_to_report("Nessun paziente ad alto rischio nel dataset", "", 'text')

# =============================================================================
# SALVATAGGIO REPORT
# =============================================================================
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Report Analisi Sanitaria</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Calibri&display=swap');
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Calibri', 'Arial', sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #e9eef5 100%);
            color: #4a5568;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        h1 {{
            font-family: 'Cambria', 'Times New Roman', serif;
            font-weight: bold;
            color: #6b8c9e;
            border-bottom: 3px solid #b3cdd9;
            padding-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
            letter-spacing: 1px;
        }}
        
        h2 {{
            font-family: 'Cambria', 'Times New Roman', serif;
            font-weight: bold;
            color: #7fa5b8;
            border-bottom: 2px solid #d0e0e8;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        
        h3 {{
            font-family: 'Cambria', 'Times New Roman', serif;
            font-weight: bold;
            color: #95b9cc;
            margin-top: 25px;
        }}
        
        .table-container {{
            width: 100%;
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            font-size: 14px;
        }}
        
        th {{
            background: linear-gradient(135deg, #b3cdd9 0%, #9fb9c9 100%);
            color: white;
            padding: 15px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
        }}
        
        td {{
            border: 1px solid #e2e8f0;
            padding: 12px;
            color: #4a5568;
            white-space: nowrap;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        
        tr:hover {{
            background-color: #e6f0f5;
            transition: background-color 0.3s ease;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        
        .chart-item {{
            background: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.05);
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .chart-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
            border-color: #b3cdd9;
        }}
        
        .chart-item img {{
            width: 100%;
            height: auto;
            border-radius: 10px;
        }}
        
        .chart-title {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
            color: #6b8c9e;
            font-size: 14px;
        }}
        
        pre {{
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #b3cdd9;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            font-family: 'Consolas', monospace;
            font-size: 14px;
            overflow-x: auto;
        }}
        
        p {{
            line-height: 1.8;
            font-size: 16px;
            color: #5f6b7a;
        }}
        
        /* Modal per ingrandimento */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            overflow: auto;
        }}
        
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
        
        .modal-content img {{
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(255,255,255,0.2);
        }}
        
        .close {{
            position: absolute;
            top: 20px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
            z-index: 1001;
        }}
        
        .close:hover {{
            color: #b3cdd9;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            th, td {{
                padding: 8px;
                font-size: 12px;
            }}
        }}
    </style>
</head>
<body>
    <div id="myModal" class="modal" onclick="closeModal(event)">
        <span class="close">&times;</span>
        <div class="modal-content">
            <img id="modalImg">
        </div>
    </div>

    <div class="container">
        {''.join(html_content)}
    </div>

    <script>
        function openModal(element) {{
            var modal = document.getElementById('myModal');
            var modalImg = document.getElementById('modalImg');
            var img = element.querySelector('img');
            modal.style.display = "block";
            modalImg.src = img.src;
        }}
        
        function closeModal(event) {{
            if (event.target.className === 'modal' || event.target.className === 'close') {{
                document.getElementById('myModal').style.display = "none";
            }}
        }}
        
        document.addEventListener('keydown', function(event) {{
            if (event.key === "Escape") {{
                document.getElementById('myModal').style.display = "none";
            }}
        }});
    </script>
</body>
</html>
"""

with open(report_file, 'w', encoding='utf-8') as f:
    f.write(html_template)

# =============================================================================
# SALVATAGGIO DATI
# =============================================================================
df.to_csv('dataset_analizzato.csv', index=False)
feature_importance.to_csv('feature_importance_regressione.csv', index=False)
feature_importance_clf.to_csv('feature_importance_classificazione.csv', index=False)
risultati.to_csv('predizioni_nuovi_pazienti.csv', index=False)

print(f"\nReport generato: {report_file}")
print("File salvati:")
print("- dataset_analizzato.csv")
print("- feature_importance_regressione.csv")
print("- feature_importance_classificazione.csv")
print("- predizioni_nuovi_pazienti.csv")