import sys
import time
import os
from server.rabbitmq.client import RabbitMQ

def start_listener(queue_name):
    rabbitmq = None

    while True:
        try:
            print("🔌 Connecting to RabbitMQ...")
            rabbitmq = RabbitMQ()
            print(f"📋 Declaring queue {queue_name}...")
            rabbitmq.channel.queue_declare(queue=queue_name, durable=True)
            print(f"👂 Listening for messages on queue {queue_name}")
            print("✅ Successfully connected and listening...")
            rabbitmq.consume(queue_name=queue_name, callback=_message_callback)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            if rabbitmq:
                rabbitmq.close()
            rabbitmq = None
            print("💤 Sleeping 5 seconds before retry...")
            time.sleep(5)

def _message_callback(ch, method, properties, body):
    """Callback function to process received messages"""
    try:
        message = body.decode('utf-8')
        print(f"📨 Received message: {message}")

        print(f"🔄 Processing document URL: {message}")

        # TODO: Add actual ingestion logic here
        # This is where you would call the ingest logic

        print(f"✅ Successfully processed message: {message}")

    except Exception as e:
        print(f"❌ Error processing message: {e}")

def main():
    """Entry point for the ingestion service"""
    print("🚀 Starting Ingestion Service...")
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Working directory: {os.getcwd()}" if hasattr(__builtins__, 'os') else "📍 Working directory: unknown")

    queue_name = "documents_to_process"

    try:
        # Brief wait since health checks ensure services are ready
        print("⏳ Waiting for services to be ready...")
        time.sleep(2)

        print(f"🌍 Environment variables:")
        print(f"  - RABBITMQ_HOST: {os.getenv('RABBITMQ_HOST', 'not set')}")
        print(f"  - RABBITMQ_PORT: {os.getenv('RABBITMQ_PORT', 'not set')}")
        print(f"  - RABBITMQ_USER: {os.getenv('RABBITMQ_USER', 'not set')}")

        print(f"👂 Starting listener for queue: {queue_name}")
        start_listener(queue_name)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down ingestion service...")
        print("👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error in ingestion service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
