# main.py
import ganim
import torch

# --- WORKFLOW DE TREINAMENTO ---
print("--- Iniciando workflow de treinamento ---")
try:
    # Apague o modelo antigo se ele foi treinado com outra configuração
    import os
    if os.path.exists('./meu_gerador.pth'):
        print("Modelo antigo encontrado. Recomenda-se apagar se a configuração mudou.")

    # 1. O usuário agora pode controlar a prévia!
    ganim.setup(
        epochs=10, 
        imageSize=64, # Lembre-se que a arquitetura atual é fixa para 64x64
        sampleInterval=5,
        previewImageCount=25,  # <-- QUERO VER 25 IMAGENS (5x5 grid)
        previewWindowSize=768  # <-- E NUMA JANELA 768x768
    )

    # 2. Treinar o modelo
    trained_generator, history = ganim.fit(data='./imgs')

    if trained_generator and history:
        ganim.plot(history)
        ganim.save(trained_generator, path='./meu_gerador.pth')
        print("\nWorkflow de treinamento concluído com sucesso!")
    else:
        print("\nWorkflow de treinamento falhou ou foi interrompido.")

except FileNotFoundError:
    print("\nERRO: A pasta './imgs' não foi encontrada ou está vazia.")
    print("Por favor, crie a pasta e coloque um subdiretório com suas imagens de treino dentro dela.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado durante o treino: {e}")


# --- WORKFLOW DE GERAÇÃO ---
print("\n--- Iniciando workflow de geração ---")
try:
    # 4. Carregar um modelo já treinado
    generator = ganim.load(path='./meu_gerador.pth')

    # 5. Gerar um número diferente de imagens
    new_images = ganim.sample(model=generator, count=4) # Pedindo 4 imagens (2x2 grid)

    # 6. Mostrar as imagens em uma janela de tamanho personalizado
    ganim.show(
        new_images, 
        window_title="Imagens Finais por Ganim",
        window_size=1024 # <-- JANELA FINAL BEM GRANDE!
    )
    print("Workflow de geração concluído!")

except FileNotFoundError:
    print("\nModelo './meu_gerador.pth' não encontrado.")
    print("Execute o workflow de treinamento primeiro para criar o modelo.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado durante a geração: {e}")