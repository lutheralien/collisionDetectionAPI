# run.py
from app import create_app
import logging
from waitress import serve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    app = create_app()
    port = app.config['PORT']
    
    if app.debug:
        logger.info(f"Starting server in development mode on port {port}...")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True
        )
    else:
        logger.info(f"Starting production server with waitress on port {port}...")
        serve(
            app,
            host='0.0.0.0',
            port=port,
            threads=4,
            connection_limit=1000,
            channel_timeout=30
        )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)