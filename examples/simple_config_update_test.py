"""
Script de teste simples para verificar se a atualização de configuração funciona via Socket.IO.

INSTRUÇÕES PARA TESTAR:
1. Terminal 1: python relay_server.py 5000
2. Terminal 2: python main_loop.py robot_1
3. Terminal 3: python examples/simple_config_update_test.py

Você verá as mensagens de atualização no console do main_loop.py
"""

import socketio
import time

# Configuração
HOST = "127.0.0.1"
PORT = 5000
ROBOT_ID = "robot_1"

print("=== Teste Simples de Atualização de Configuração ===\n")
print("Conectando ao relay server...")

sio = socketio.Client()

@sio.event
def connect():
    print(f"✓ Conectado ao relay server em {HOST}:{PORT}\n")

try:
    sio.connect(f'http://{HOST}:{PORT}')
    time.sleep(0.5)
    
    print("Enviando atualização: strength=350, safety_margin=75\n")
    
    message = {
        'topic': 'Neuronavigation to Robot: Update repulsion field config',
        'data': {
            'robot_ID': ROBOT_ID,
            'config_updates': {
                'strength': 350,
                'safety_margin': 75
            }
        }
    }
    
    sio.emit('from_neuronavigation', message)
    print("✓ Mensagem enviada!")
    print("\nVerifique o console do main_loop.py para ver a confirmação da atualização.")
    
    time.sleep(2)
    
    sio.disconnect()
    print("\n✓ Teste concluído!")

except Exception as e:
    print(f"✗ Erro: {e}")
    print("\nCertifique-se de que:")
    print("  1. O relay server está rodando: python relay_server.py 5000")
    print("  2. O main_loop está rodando: python main_loop.py robot_1")
