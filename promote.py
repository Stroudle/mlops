import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

# Modelos a serem avaliados
model_names = ["K-NearestNeighborsGridSearch", "LinearRegressionGridSearch"]

# Definir os limites de R-squared score para Staging e Production
staging_threshold = 0.8

best_overall_model = None
best_overall_r2_score = 0
best_overall_model_name = None

for model_name in model_names:
    print(f"\nAvaliando versões do modelo: {model_name}")

    versions = client.search_model_versions(f"name='{model_name}'")

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics

        if "r2" in metrics:
            r2 = metrics["r2"]

            # Promover para Staging se o R-squared for maior que o limite
            if r2 > staging_threshold:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Staging"
                )
                print(f"Modelo {model_name} versão {version.version} com R-squared score {r2} movido para Staging.")

            # Atualizar o melhor modelo geral
            if r2 > best_overall_r2_score:
                best_overall_r2_score = r2
                best_overall_model = version.version
                best_overall_model_name = model_name

# Promover o melhor modelo geral para Production
if best_overall_model:
    client.transition_model_version_stage(
        name=best_overall_model_name,
        version=best_overall_model,
        stage="Production"
    )
    print(f"\nO melhor modelo geral é {best_overall_model_name} versão {best_overall_model} com R-squared score {best_overall_r2_score}, promovido para Production.")
else:
    print("\nNenhum modelo atende ao critério para ser promovido.")