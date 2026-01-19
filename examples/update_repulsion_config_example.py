"""
Exemplo de como atualizar dinamicamente a configuração do RepulsionField durante a execução.

Este script demonstra duas formas de atualizar os parâmetros:
1. Através de mensagens pubsub (integrado com o sistema existente)
2. Através de um exemplo de loop interativo
"""

from robot import pub
from robot.constants import PUB_MESSAGES, FUNCTION_UPDATE_REPULSION_CONFIG


def send_config_update(config_updates):
    """
    Envia uma atualização de configuração para todas as instâncias de RepulsionField.
    
    Args:
        config_updates (dict): Dicionário com as chaves de configuração a atualizar.
                              Chaves válidas: 'strength', 'safety_margin', 'ema', 'stop_distance'
    
    Example:
        send_config_update({'strength': 300, 'safety_margin': 70})
    """
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates=config_updates
    )


if __name__ == "__main__":
    print("=== Exemplo de Atualização Dinâmica de Configuração do RepulsionField ===\n")
    
    # Exemplo 1: Atualizar força de repulsão
    print("1. Aumentando a força de repulsão para 300:")
    send_config_update({'strength': 300})
    
    input("Pressione Enter para continuar...")
    
    # Exemplo 2: Atualizar margem de segurança
    print("\n2. Aumentando margem de segurança para 80mm:")
    send_config_update({'safety_margin': 80})
    
    input("Pressione Enter para continuar...")
    
    # Exemplo 3: Atualizar múltiplos parâmetros de uma vez
    print("\n3. Atualizando múltiplos parâmetros:")
    send_config_update({
        'strength': 200,
        'safety_margin': 60,
        'ema': 0.15,
        'stop_distance': 8
    })
    
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
                
                send_config_update({param: value})
                
            except ValueError:
                print(f"Valor inválido: {value}")
                
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário.")
            break
        except Exception as e:
            print(f"Erro: {e}")
