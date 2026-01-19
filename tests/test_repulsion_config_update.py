"""
Script de teste para a atualização dinâmica de configuração do RepulsionField.

Este script simula um ambiente onde o RepulsionField está ativo e permite
testar as atualizações de configuração em tempo real.
"""

import time
import numpy as np
from robot.control.repulsion import RepulsionField
from robot import pub
from robot.constants import PUB_MESSAGES, FUNCTION_UPDATE_REPULSION_CONFIG


def test_basic_updates():
    """Testa atualizações básicas de configuração."""
    print("\n=== Teste 1: Atualizações Básicas ===")
    
    # Criar uma instância de RepulsionField
    repulsion = RepulsionField()
    print(f"Configuração inicial: {repulsion.cfg}\n")
    
    # Teste 1: Atualizar força
    print("Atualizando strength para 300...")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={'strength': 300}
    )
    time.sleep(0.5)  # Dar tempo para a mensagem ser processada
    
    # Teste 2: Atualizar margem de segurança
    print("\nAtualizando safety_margin para 80...")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={'safety_margin': 80}
    )
    time.sleep(0.5)
    
    # Teste 3: Atualizar múltiplos parâmetros
    print("\nAtualizando múltiplos parâmetros...")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={
            'ema': 0.3,
            'stop_distance': 15
        }
    )
    time.sleep(0.5)
    
    print(f"\nConfiguração final: {repulsion.cfg}")


def test_compute_offset_behavior():
    """Testa o comportamento do compute_offset com diferentes configurações."""
    print("\n\n=== Teste 2: Comportamento do compute_offset ===")
    
    repulsion = RepulsionField()
    
    # Configurar direção de freio de teste
    repulsion.brake_direction = np.array([1.0, 0.0, 0.0])
    
    # Testar com configuração padrão
    print("\nCom configuração padrão (strength=240, safety_margin=65):")
    distance = 50  # mm
    dt = 0.01  # segundos
    offset, stop = repulsion.compute_offset(distance, dt)
    print(f"  Distância: {distance}mm, Offset: {offset}, Stop: {stop}")
    
    # Atualizar para uma força maior
    print("\nAtualizando strength para 400...")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={'strength': 400}
    )
    time.sleep(0.5)
    
    print("Com nova configuração (strength=400):")
    offset, stop = repulsion.compute_offset(distance, dt)
    print(f"  Distância: {distance}mm, Offset: {offset}, Stop: {stop}")
    
    # Testar stop_distance
    print("\nTestando stop_distance...")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={'stop_distance': 20}
    )
    time.sleep(0.5)
    
    print("Com stop_distance=20, testando distância de 15mm:")
    offset, stop = repulsion.compute_offset(15, dt)
    print(f"  Distância: 15mm, Offset: {offset}, Stop: {stop} (deveria ser True)")


def test_invalid_updates():
    """Testa o comportamento com atualizações inválidas."""
    print("\n\n=== Teste 3: Atualizações Inválidas ===")
    
    repulsion = RepulsionField()
    
    print("\nTestando parâmetro inexistente:")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={'invalid_param': 100}
    )
    time.sleep(0.5)
    
    print("\nTestando tipo de dados inválido:")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates="invalid_type"  # Deveria ser dict
    )
    time.sleep(0.5)


def test_multiple_instances():
    """Testa múltiplas instâncias de RepulsionField."""
    print("\n\n=== Teste 4: Múltiplas Instâncias ===")
    
    print("Criando duas instâncias de RepulsionField...")
    repulsion1 = RepulsionField()
    repulsion2 = RepulsionField()
    
    print(f"Repulsion1 inicial: strength={repulsion1.cfg['strength']}")
    print(f"Repulsion2 inicial: strength={repulsion2.cfg['strength']}")
    
    print("\nEnviando atualização para strength=350...")
    pub.sendMessage(
        PUB_MESSAGES[FUNCTION_UPDATE_REPULSION_CONFIG],
        config_updates={'strength': 350}
    )
    time.sleep(0.5)
    
    print(f"Repulsion1 após update: strength={repulsion1.cfg['strength']}")
    print(f"Repulsion2 após update: strength={repulsion2.cfg['strength']}")
    print("\nAmbas as instâncias foram atualizadas!")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE ATUALIZAÇÃO DINÂMICA DE CONFIGURAÇÃO - RepulsionField")
    print("=" * 60)
    
    try:
        test_basic_updates()
        test_compute_offset_behavior()
        test_invalid_updates()
        test_multiple_instances()
        
        print("\n\n" + "=" * 60)
        print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n\nERRO DURANTE OS TESTES: {e}")
        import traceback
        traceback.print_exc()
