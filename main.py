import os

import ganim


def run_training_workflow():
    print("--- Iniciando workflow de treinamento ---")
    try:
        if os.path.exists("./meu_gerador.pth"):
            print("Modelo antigo encontrado. Recomenda-se apagar se a configuração mudou.")

        ganim.setup(
            epochs=10,
            image_size=64,
            sample_interval=5,
            preview_image_count=25,
            preview_window_size=768,
        )

        trained_generator, history = ganim.fit(data="./data")

        if trained_generator and history:
            ganim.plot(history)
            ganim.save(trained_generator, path="./meu_gerador.pth")
            print("\nWorkflow de treinamento concluído com sucesso!")
            return True

        print("\nWorkflow de treinamento falhou ou foi interrompido.")
        return False
    except FileNotFoundError:
        print("\nERRO: A pasta './data' não foi encontrada ou está vazia.")
        print("Por favor, crie a pasta e coloque um subdiretório com suas imagens de treino dentro dela.")
        return False
    except Exception as exc:
        print(f"\nOcorreu um erro inesperado durante o treino: {exc}")
        return False


def run_generation_workflow():
    print("\n--- Iniciando workflow de geração ---")
    try:
        generator = ganim.load(path="./meu_gerador.pth")
        new_images = ganim.sample(model=generator, count=4)
        ganim.show(
            new_images,
            window_title="Imagens Finais por Ganim",
            window_size=1024,
        )
        print("Workflow de geração concluído!")
        return True
    except FileNotFoundError:
        print("\nModelo './meu_gerador.pth' não encontrado.")
        print("Execute o workflow de treinamento primeiro para criar o modelo.")
        return False
    except Exception as exc:
        print(f"\nOcorreu um erro inesperado durante a geração: {exc}")
        return False


if __name__ == "__main__":
    run_training_workflow()
    run_generation_workflow()
