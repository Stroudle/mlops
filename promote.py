import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
model_name = "RandomForestGridSearch"

# Definir os limites de R-squared score para Staging e Production
staging_threshold = 0.8

versions = client.search_model_versions(f"name='{model_name}'")

best_model = None
best_r2_score = 0

for version in versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics

    if "r2" in metrics:
        r2 = metrics["r2"]

        if r2 > staging_threshold:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Staging"
            )
            print(f"Modelo versão {version.version} com R-squared score {r2} movido para Staging.")

        if r2 > best_r2_score:
            best_r2_score = r2
            best_model = version.version

if best_model:
    client.transition_model_version_stage(
        name=model_name,
        version=best_model,
        stage="Production"
    )
    print(f"Modelo versão {best_model} agora é o Champion com R-squared score {best_r2_score}.")
else:
    print("Nenhum modelo atende ao critério para ser Champion.")