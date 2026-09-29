"""
Extract quantifiable NLP features from daily aligned news headlines and metadata. Strictly adheres to classical NLP: Domain-adapted Lexicons, Sublinear TF-IDF + LSA, and Macroeconomic/Geopolitical Entity Extraction.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.config import data_path, load_config

# Domain-specific bilingual financial lexicons
ID_FINANCIAL_LEXICON = {
    'positive': [
        'menguat', 'penguatan', 'melonjak', 'surplus', 'naik', 'kenaikan', 'tumbuh', 'pertumbuhan',
        'rebound', 'moncer', 'untung', 'laba', 'positif', 'stabil', 'membaik', 'optimis',
        'perkasa', 'berjaya', 'melaju', 'terkerek', 'borong', 'rekor', 'bergairah', 'hijau',
        'bullish', 'terangkat', 'cemerlang', 'melambung', 'menanjak', 'rally', 'gain'
    ],
    'negative': [
        'melemah', 'pelemahan', 'anjlok', 'merosot', 'defisit', 'turun', 'penurunan', 'terkoreksi',
        'koreksi', 'loyo', 'rugi', 'negatif', 'ambrol', 'tertekan', 'pesimis', 'terpuruk',
        'rontok', 'jatuh', 'tumbang', 'ambruk', 'merah', 'bearish', 'inflasi',
        'krisis', 'resesi', 'gagal bayar', 'pailit', 'bengkak', 'slump', 'deficit'
    ],
    'uncertainty': [
        'ketidakpastian', 'volatilitas', 'fluktuatif', 'waspada', 'bayang-bayang', 'risiko',
        'was-was', 'teka-teki', 'ancaman', 'menanti', 'mencermati', 'antisipasi', 'dilema',
        'wait and see', 'tapering', 'gejolak', 'goncangan', 'imbas', 'rentan', 'uncertainty'
    ],
    'tightening_hawkish': [
        'kenaikan suku bunga', 'suku bunga naik', 'pengetatan', 'hawkish', 'tapering',
        'kerek suku bunga', 'the fed naikkan', 'inflasi as', 'dolar perkasa', 'yield us treasury'
    ]
}

ENTITY_KEYWORDS = {
    'ent_fed': ['the fed', 'fed', 'powell', 'jerome powell', 'fomc'],
    'ent_bi': ['bank indonesia', 'bi ', 'perry warjiyo', 'rdg bi'],
    'ent_usd': ['dolar as', 'dolar', 'greenback', 'usd'],
    'ent_rupiah': ['rupiah', 'idr', 'kurs rupiah', 'nilai tukar'],
    'ent_inflation': ['inflasi', 'cpi', 'ihk'],
    'ent_rate': ['suku bunga', 'bi rate', 'ffr', 'fed rate', 'bunga acuan'],
    'ent_geopolitics': ['perang', 'rusia', 'ukraina', 'taiwan', 'china', 'tiongkok', 'sanksi', 'konflik', 'rudal', 'militer', 'taliban'],
    'ent_trade_cadangan': ['neraca perdagangan', 'ekspor', 'impor', 'cadangan devisa', 'surplus neraca']
}

INDONESIAN_STOPWORDS = [
    'di', 'dan', 'ke', 'dari', 'ini', 'itu', 'pada', 'untuk', 'yang', 'dengan', 'oleh',
    'dalam', 'adalah', 'akan', 'juga', 'tak', 'tidak', 'bisa', 'ada', 'lagi', 'saat',
    'jadi', 'atas', 'karena', 'hingga', 'setelah', 'terhadap', 'soal', 'apa', 'kabar',
    'bisnis', 'kontan', 'com', 'read', 'news', 'market', 'nasional', 'ekonomi',
    'investasi', 'keuangan', 'hari', 'siap', 'catat', 'cek', 'simak', 'bocoran', 'buka'
]


def extract_lexicon_features(text: str) -> dict:
    """Compute dictionary-based financial sentiment & uncertainty ratios."""
    if not isinstance(text, str) or not text.strip():
        return {
            'lex_pos_count': 0, 'lex_neg_count': 0, 'lex_unc_count': 0, 'lex_tight_count': 0,
            'lex_pos_ratio': 0.0, 'lex_neg_ratio': 0.0, 'lex_net_polarity': 0.0, 'lex_uncertainty_ratio': 0.0
        }
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = max(len(words), 1)

    pos_count = sum(text_lower.count(w) for w in ID_FINANCIAL_LEXICON['positive'])
    neg_count = sum(text_lower.count(w) for w in ID_FINANCIAL_LEXICON['negative'])
    unc_count = sum(text_lower.count(w) for w in ID_FINANCIAL_LEXICON['uncertainty'])
    tight_count = sum(text_lower.count(w) for w in ID_FINANCIAL_LEXICON['tightening_hawkish'])

    return {
        'lex_pos_count': pos_count,
        'lex_neg_count': neg_count,
        'lex_unc_count': unc_count,
        'lex_tight_count': tight_count,
        'lex_pos_ratio': pos_count / total_words,
        'lex_neg_ratio': neg_count / total_words,
        'lex_net_polarity': (pos_count - neg_count) / (pos_count + neg_count + 1e-5),
        'lex_uncertainty_ratio': unc_count / total_words
    }


def extract_entity_features(text: str) -> dict:
    """Extract occurrence frequencies of key macroeconomic and geopolitical entities."""
    if not isinstance(text, str) or not text.strip():
        return {k: 0 for k in ENTITY_KEYWORDS}
    text_lower = text.lower()
    return {k: sum(text_lower.count(kw) for kw in kws) for k, kws in ENTITY_KEYWORDS.items()}


def build_nlp_features():
    config = load_config()
    processed_dir = Path(config['paths']['processed_dir'])
    dataset_path = processed_dir / 'final_dataset.csv'

    if not dataset_path.exists():
        raise SystemExit(f'{dataset_path} not found. Complete Task 1 alignment first.')

    df = pd.read_csv(dataset_path)
    print(f'Ingested {len(df)} trading day rows from {dataset_path.name}')

    titles_series = df['titles'].fillna('')

    # 1. Domain-adapted financial lexicon scoring
    print('Extracting financial lexicon and uncertainty features...')
    lex_df = pd.DataFrame([extract_lexicon_features(t) for t in titles_series])

    # 2. Institutional entity frequencies
    print('Extracting macroeconomic and geopolitical entity counts...')
    ent_df = pd.DataFrame([extract_entity_features(t) for t in titles_series])

    # 3. Sublinear TF-IDF + Latent Semantic Analysis (TruncatedSVD)
    print('Fitting classical TF-IDF (1, 2-grams) and TruncatedSVD (15 components)...')
    tfidf = TfidfVectorizer(
        stop_words=INDONESIAN_STOPWORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        sublinear_tf=True,
        max_features=2500
    )
    tfidf_matrix = tfidf.fit_transform(titles_series)

    svd = TruncatedSVD(n_components=15, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    lsa_cols = [f'lsa_topic_{i+1}' for i in range(15)]
    lsa_df = pd.DataFrame(lsa_matrix, columns=lsa_cols)

    # 4. News volume and GDELT sentiment metadata
    tone_clean = df['gdelt_tone_mean'].fillna(0.0)
    has_news_flag = (df['n_articles'] > 0).astype(int)

    meta_nlp_df = pd.DataFrame({
        'news_exists_flag': has_news_flag,
        'gdelt_tone_mean_clean': tone_clean,
        'headline_word_count': titles_series.apply(lambda s: len(re.findall(r'\b\w+\b', s))),
        'news_intensity_per_day': df['n_articles'] / (df['calendar_days_since_prev'].fillna(1).replace(0, 1))
    })

    # Assemble complete dataset
    features_df = pd.concat([df, lex_df, ent_df, lsa_df, meta_nlp_df], axis=1)

    out_path = processed_dir / 'dataset_with_nlp_features.csv'
    features_df.to_csv(out_path, index=False)
    print(f'Successfully exported dataset with {features_df.shape[1]} total columns to {out_path}')


if __name__ == '__main__':
    build_nlp_features()