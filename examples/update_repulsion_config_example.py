"""
Exemplo de como atualizar dinamicamente a configuração do RepulsionField durante a execução usando Socket.IO.

Este script conecta ao relay server e envia mensagens para atualizar os parâmetros do RepulsionField.

USO:
    1. Inicie o relay server: python relay_server.py 5000
    2. Inicie o main_loop: python main_loop.py robot_1
    3. Execute este script: python examples/update_repulsion_config_example.py
"""

import socketio
import time

# Configuração
RELAY_SERVER_HOST = "127.0.0.1"
RELAY_SERVER_PORT = 5000
ROBOT_ID = "robot_1"  # Deve corresponder ao robot_id usado no main_loop

# Mensagem que será roteada
MESSAGE_TOPIC = "Neuronavigation to Robot: Update repulsion field config"


def send_config_update(sio, config_updates):
    """
    Envia uma atualização de configuração para o RepulsionField via Socket.IO.
    
    Args:
        sio: Socket.IO client instance
        config_updates (dict): Dicionário com as chaves de configuração a atualizar.
                              Chaves válidas: 'strength', 'safety_margin', 'ema', 'stop_distance'
    
    Example:
        send_config_update(sio, {'strength': 300, 'safety_margin': 70})
    """
    message = {
        'topic': MESSAGE_TOPIC,
        'data': {
            'robot_ID': ROBOT_ID,
            'config_updates': config_updates
        }
    }
    
    sio.emit('from_neuronavigation', message)
    print(f"Sent: {config_updates}")


def main():
    print("=== Exemplo de Atualização Dinâmica de Configuração do RepulsionField ===\n")
    
    # Criar cliente Socket.IO
    sio = socketio.Client()
    
    @sio.event
    def connect():
        print(f"Conectado ao relay server em {RELAY_SERVER_HOST}:{RELAY_SERVER_PORT}\n")
    
    @sio.event
    def disconnect():
        print("Desconectado do relay server")
    
    # Conectar ao relay server
    try:
        url = f'http://{RELAY_SERVER_HOST}:{RELAY_SERVER_PORT}'
        sio.connect(url)
        time.sleep(0.5)  # Aguardar conexão
        
    except Exception as e:
        print(f"Erro ao conectar ao relay server: {e}")
        print("Certifique-se de que o relay server está rodando:")
        print(f"  python relay_server.py {RELAY_SERVER_PORT}")
        return
    
    try:
        # Exemplo 1: Atualizar força de repulsão
        print("1. Aumentando a força de repulsão para 300:")
        send_config_update(sio, {'strength': 300})
        time.sleep(1)
        
        input("Pressione Enter para continuar...")
        
        # Exemplo 2: Atualizar margem de segurança
        print("\n2. Aumentando margem de segurança para 80mm:")
        send_config_update(sio, {'safety_margin': 80})
        time.sleep(1)
        
        input("Pressione Enter para continuar...")
        
        # Exemplo 3: Atualizar múltiplos parâmetros de uma vez
        print("\n3. Atualizando múltiplos parâmetros:")
        send_config_update(sio, {
            'strength': 200,
            'safety_margin': 60,
            'ema': 0.15,
            'stop_distance': 8
        })
        time.sleep(1)
        
        input("Pressione Enter para continuar...")
        
        # Exemplo 4: Loop interativo
        print("\n4. Modo interativo - Digite os comandos ou 'sair' para terminar:")
        print("Comandos disponíveis:")
        print("  strength <valor>       - Ex: strength 250")
        print("  safety_margin <valor>  - Ex: safety_margin 70")
        print("  ema <valor>            - Ex: ema 0.2")
        print("  stop_distance <valor>  - Ex: stop_distance 12")
        print("  sair                   - Sair do modo interativo\n")
        
        while True:
            try:
                cmd = input(">>> ").strip()
                
                if cmd.lower() == 'sair':
                    print("Encerrando modo interativo.")
                    break
                
                if not cmd:
                    continue
                
                parts = cmd.split()
                if len(parts) != 2:
                    print("Formato inválido. Use: <parametro> <valor>")
                    continue
                
                param, value = parts
                
                # Converter valor para o tipo apropriado
                try:
                    if param in ['strength', 'safety_margin', 'stop_distance']:
                        value = float(value)
                    elif param == 'ema':
                        value = float(value)
                        if not 0 <= value <= 1:
                            print("O valor de 'ema' deve estar entre 0 e 1")
                            continue
                    else:
                        print(f"Parâmetro desconhecido: {param}")
                        print("Parâmetros válidos: strength, safety_margin, ema, stop_distance")
                        continue
                    
                    send_config_update(sio, {param: value})
                    time.sleep(0.5)
                    
                except ValueError:
                    print(f"Valor inválido: {value}")
                    
            except KeyboardInterrupt:
                print("\n\nInterrompido pelo usuário.")
                break
            except Exception as e:
                print(f"Erro: {e}")
    
    finally:
        # Desconectar
        if sio.connected:
            sio.disconnect()
        print("\nDesconectado.")


if __name__ == "__main__":
    main()
