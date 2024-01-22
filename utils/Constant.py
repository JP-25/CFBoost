DATASETS = []
MODELS = []
# amazon movies and TVs, musics, fashion, CDs

DATASET_TO_TYPE = {
    'ml-100k': 'UIRT',
    'ml-1m': 'UIRT',
    'ml-10m': 'UIRT',
    'ml-20m': 'UIRT',
    'netflix': 'UIRT',
    'amusic': 'UIRT',
    'yelp': 'UIRT',
    'agames': 'UIRT',
    'epinion': 'UIRT',
    'kindle_store': 'UIRT',
    'amazon_cds': 'UIRT',
    'amazon': 'UIRT',
    'gowalla': 'UIRT',
    'douban_movie': 'UIRT',
    'douban_book': 'UIRT',
    'douban_music': 'UIRT',
    'lastfm': 'UIRT',
    'amazon_grocery': 'UIRT',
    'kuairec': 'UIRT',
    'ciao': 'UIRT'
}

DATASET_TO_SEPRATOR = {
    'ml-100k': '\t',
    'ml-1m': '::',
    'ml-10m': '::',
    'ml-20m': ',',
    'netflix': ',',
    'amusic': ',',
    'yelp': '\t',
    'agames': ',',
    'epinion': ',',
    'kindle_store': ',',
    'amazon_cds': ',',
    'amazon': ',',
    'gowalla': '\t',
    'douban_movie': '\t',
    'douban_book': '\t',
    'douban_music': '\t',
    'lastfm': '\t',
    'amazon_grocery': ',',
    'kuairec': ',',
    'ciao': ','
}