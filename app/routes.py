from app import app
from flask import render_template, request, jsonify, current_app
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timedelta
import json
from cachetools import TTLCache, LRUCache
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuración de caché más eficiente
MODEL_CACHE = TTLCache(maxsize=100, ttl=300)  # 5 minutos TTL
DETAILS_CACHE = LRUCache(maxsize=1000)  # LRU cache para detalles
BATCH_SIZE = 5  # Número de modelos a procesar en paralelo

class ModelService:
    @staticmethod
    def get_model_details(model_id):
        """Obtener detalles del modelo con manejo de errores mejorado"""
        if model_id in DETAILS_CACHE:
            return DETAILS_CACHE[model_id]

        try:
            response = requests.get(
                f"https://huggingface.co/api/models/{model_id}",
                timeout=5  # Timeout para evitar bloqueos
            )
            if response.status_code == 200:
                data = response.json()
                size = ModelService._calculate_model_size(data)
                processed_data = {
                    'size': size,
                    'tags': data.get('tags', []),
                    'pipeline_tag': data.get('pipeline_tag', 'N/A'),
                    'lastModified': data.get('lastModified', 'N/A'),
                    'downloads': data.get('downloads', 0)
                }
                DETAILS_CACHE[model_id] = processed_data
                return processed_data
        except Exception as e:
            current_app.logger.error(f"Error fetching model {model_id}: {str(e)}")
        return None

    @staticmethod
    def _calculate_model_size(data):
        """Calcula el tamaño total del modelo"""
        try:
            if 'siblings' in data:
                return sum(sibling.get('size', 0) for sibling in data['siblings'])
            return data.get('size', 0)
        except Exception:
            return 0

    @staticmethod
    def process_models_batch(models):
        """Procesa un lote de modelos en paralelo"""
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {
                executor.submit(ModelService.get_model_details, model['id']): model 
                for model in models
            }
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    details = future.result()
                    if details:
                        model.update(details)
                except Exception as e:
                    current_app.logger.error(f"Error processing model {model['id']}: {str(e)}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/models')
def list_models():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 12, type=int)
    cache_key = f"models_page_{page}_limit_{limit}"

    if cache_key in MODEL_CACHE:
        return render_template('models.html', **MODEL_CACHE[cache_key])

    try:
        models = fetch_models(page, limit)
        ModelService.process_models_batch(models)
        
        template_data = {
            'models': models,
            'current_page': page,
            'limit': limit
        }
        MODEL_CACHE[cache_key] = template_data
        return render_template('models.html', **template_data)
    except Exception as e:
        current_app.logger.error(f"Error listing models: {str(e)}")
        return render_template('error.html', error="Error cargando modelos"), 500

def fetch_models(page, limit):
    """Función centralizada para obtener modelos de la API"""
    offset = (page - 1) * limit
    response = requests.get(
        'https://huggingface.co/api/models',
        params={
            'limit': limit,
            'offset': offset,
            'sort': 'downloads',
            'direction': -1,
            'filter': 'text-generation'
        },
        timeout=5
    )
    return response.json()

@app.route('/load_more_models')
def load_more_models():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 12, type=int)
    offset = (page - 1) * limit
    
    response = requests.get(
        'https://huggingface.co/api/models',
        params={
            'limit': limit,
            'offset': offset,
            'sort': 'downloads',
            'direction': -1,
            'filter': 'text-generation'  # Filtrar solo modelos de texto
        }
    )
    models = response.json()
    ModelService.process_models_batch(models)
    
    return jsonify(models)

@app.route('/run_model', methods=['POST'])
def run_model():
    model_id = request.form['model_id']
    # Lógica para ejecutar el modelo localmente
    return f"Running model {model_id}"

@app.route('/search_models')
def search_models():
    query = request.args.get('query', '')
    cache_key = f"search_{query}"
    
    if cache_key in MODEL_CACHE:
        return jsonify(MODEL_CACHE[cache_key])

    try:
        response = requests.get(
            'https://huggingface.co/api/models',
            params={
                'search': query,
                'filter': 'text-generation',
                'sort': 'downloads',
                'direction': -1,
                'limit': 12
            },
            timeout=5
        )
        models = response.json()
        ModelService.process_models_batch(models)
        MODEL_CACHE[cache_key] = models
        return jsonify(models)
    except Exception as e:
        current_app.logger.error(f"Error searching models: {str(e)}")
        return jsonify({"error": "Error searching models"}), 500
