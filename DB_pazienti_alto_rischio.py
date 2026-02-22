import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_database_from_csv():
    """Crea database SQLite da file CSV esistenti"""
    
    # =============================================================================
    # CONNESSIONE DATABASE
    # =============================================================================
    conn = sqlite3.connect('database_sanitario.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # =============================================================================
    # LETTURA FILE CSV
    # =============================================================================
    print("LETTURA FILE CSV")
    print("="*60)
    
    pazienti_df = pd.read_csv('pazienti.csv')
    alimentazione_df = pd.read_csv('alimentazione.csv')
    esami_df = pd.read_csv('esami_clinici.csv')
    
    print(f"pazienti.csv: {len(pazienti_df)} record")
    print(f"alimentazione.csv: {len(alimentazione_df)} record")
    print(f"esami_clinici.csv: {len(esami_df)} record")
    
    # =============================================================================
    # CREAZIONE TABELLE
    # =============================================================================
    print("\nCREAZIONE TABELLE")
    print("="*60)
    
    cursor.execute("DROP TABLE IF EXISTS esami_clinici")
    cursor.execute("DROP TABLE IF EXISTS alimentazione")
    cursor.execute("DROP TABLE IF EXISTS pazienti")
    
    cursor.execute('''
    CREATE TABLE pazienti (
        id_paziente TEXT PRIMARY KEY,
        nome TEXT NOT NULL,
        cognome TEXT NOT NULL,
        eta INTEGER,
        sesso TEXT,
        fumatore TEXT,
        attivita_fisica TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE alimentazione (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_paziente TEXT,
        data DATE,
        grassi_saturi_g REAL,
        zuccheri_g REAL,
        calorie_totali INTEGER,
        FOREIGN KEY (id_paziente) REFERENCES pazienti(id_paziente)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE esami_clinici (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_paziente TEXT,
        data_esame DATE,
        colesterolo_totale INTEGER,
        HDL INTEGER,
        LDL INTEGER,
        glicemia INTEGER,
        FOREIGN KEY (id_paziente) REFERENCES pazienti(id_paziente)
    )
    ''')
    
    print("Tabelle create")
    
    # =============================================================================
    # CARICAMENTO DATI
    # =============================================================================
    print("\nCARICAMENTO DATI")
    print("="*60)
    
    pazienti_df.to_sql('pazienti', conn, if_exists='append', index=False)
    print(f"Caricati {len(pazienti_df)} pazienti")
    
    alimentazione_df.to_sql('alimentazione', conn, if_exists='append', index=False)
    print(f"Caricati {len(alimentazione_df)} record alimentazione")
    
    esami_df.to_sql('esami_clinici', conn, if_exists='append', index=False)
    print(f"Caricati {len(esami_df)} record esami clinici")
    
    conn.commit()
    
    # =============================================================================
    # VERIFICA INTEGRITA'
    # =============================================================================
    print("\nVERIFICA DATABASE")
    print("="*60)
    
    for table in ['pazienti', 'alimentazione', 'esami_clinici']:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} record")
    
    # =============================================================================
    # ANALISI DATI
    # =============================================================================
    print("\nANALISI DATI")
    print("="*60)
    
    # Pazienti con colesterolo alto
    query1 = '''
    SELECT p.nome, p.cognome, p.eta, e.colesterolo_totale, e.LDL, e.glicemia
    FROM pazienti p
    JOIN esami_clinici e ON p.id_paziente = e.id_paziente
    WHERE e.colesterolo_totale > 200
    ORDER BY e.colesterolo_totale DESC
    '''
    result1 = pd.read_sql(query1, conn)
    print("\nPazienti con colesterolo > 200:")
    print(result1.to_string(index=False))
    
    # Medie alimentari per paziente
    query2 = '''
    SELECT 
        p.id_paziente,
        p.nome,
        p.cognome,
        ROUND(AVG(a.grassi_saturi_g), 1) as media_grassi,
        ROUND(AVG(a.zuccheri_g), 1) as media_zuccheri,
        e.colesterolo_totale
    FROM pazienti p
    LEFT JOIN alimentazione a ON p.id_paziente = a.id_paziente
    LEFT JOIN esami_clinici e ON p.id_paziente = e.id_paziente
    GROUP BY p.id_paziente
    ORDER BY e.colesterolo_totale DESC
    '''
    result2 = pd.read_sql(query2, conn)
    print("\nMedie alimentari e colesterolo:")
    print(result2.to_string(index=False))
    
    # Correlazione età - grassi - colesterolo
    query3 = '''
    SELECT 
        p.eta,
        AVG(a.grassi_saturi_g) as media_grassi,
        AVG(e.colesterolo_totale) as media_colesterolo
    FROM pazienti p
    JOIN alimentazione a ON p.id_paziente = a.id_paziente
    JOIN esami_clinici e ON p.id_paziente = e.id_paziente
    GROUP BY p.eta
    ORDER BY p.eta
    '''
    result3 = pd.read_sql(query3, conn)
    print("\nCorrelazione età - grassi - colesterolo:")
    print(result3.to_string(index=False))
    
    conn.close()
    print("\nDatabase creato con successo")

if __name__ == "__main__":
    create_database_from_csv()