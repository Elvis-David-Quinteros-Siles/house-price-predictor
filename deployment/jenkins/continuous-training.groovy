// Entrenamiento continuo del House Price Predictor (job house-price-ct).
//
// El job de Jenkins solo hace `load` de este archivo desde el despliegue
// (~/house-price-predictor/app, montado en Jenkins en la misma ruta): ver
// deployment/jenkins/README.md. Lo dispara house-price-release-sales al llegar
// ventas nuevas, y un cron diario por si saltó la alerta de drift.
//
//   ¿Datos nuevos? ── no ──> ¿drift? ── sí ──> INESTABLE: sin precios reales nuevos
//        │ sí                                  reentrenar daría el mismo modelo
//        v
//   error real del champion con las ventas nuevas (MLflow + /metrics)
//        v
//   entrenar -> puerta de calidad -> champion vs challenger
//        │ gana                          │ pierde: INESTABLE, sigue el champion
//        v
//   reiniciar la API -> comprobar que sirve el nuevo @champion
//
// Avisa por correo (scripts/notify.py) de todo lo que no es "sin cambios":
// promoción, challenger perdido, drift sin datos y fallos del ciclo.
properties([
    disableConcurrentBuilds(),
    buildDiscarder(logRotator(numToKeepStr: '60')),
    pipelineTriggers([cron('H 6 * * *')]),
])

// Códigos de salida de scripts/continuous_training.py y scripts/train.sh.
YES = 0
NO = 10
CHALLENGER_LOST = 3
DEGRADED = 4  // scripts/production_performance.py

def compose(String args, boolean status = false) {
    sh(script: "cd /home/ubuntu/house-price-predictor && docker compose ${args}", returnStatus: status)
}

// Un aviso que no sale no tumba el ciclo: queda anotado en el log.
def notify(String event, String detail) {
    withEnv(["NOTIFY_DETAIL=${detail}"]) {
        def rc = sh(script: 'cd /home/ubuntu/house-price-predictor && docker compose run --rm --no-deps -T '
                            + "-e NOTIFY_DETAIL -e BUILD_URL trainer python scripts/notify.py ${event}",
                    returnStatus: true)
        if (rc != 0) {
            echo "No se pudo enviar el aviso ${event} (exit ${rc})"
        }
    }
}

def check(String label, int rc) {
    if (rc != YES && rc != NO) {
        error("${label}: exit ${rc}")
    }
    rc == YES
}

timestamps {
  try {
    def newData = false
    def drift = false
    def degraded = false

    stage('Datos nuevos') {
        newData = check('new-data', compose('run --rm trainer python scripts/continuous_training.py new-data', true))
    }

    stage('Drift') {
        // El entrenador no está en la red de Prometheus: el mismo script en un
        // contenedor suelto de la imagen, en core-monitoring-net.
        drift = check('drift', sh(script: 'docker run --rm --network core-monitoring-net house-price-trainer:latest '
                                         + 'python scripts/continuous_training.py drift', returnStatus: true))
    }

    if (!newData) {
        if (drift) {
            unstable('Drift activo pero sin ventas nuevas con precio: reentrenar daría el mismo modelo. '
                     + 'Revisar qué casas se tasan (Grafana -> House Price Predictor).')
            currentBuild.description = 'drift sin datos nuevos'
            notify('drift_no_data', currentBuild.description)
        } else {
            currentBuild.description = 'sin cambios'
        }
        return
    }

    stage('Error real') {
        // El champion tasa las ventas que acaban de llegar, antes de que el
        // reentrenamiento las aprenda: es su error en producción.
        def rc = compose('run --rm trainer python scripts/production_performance.py', true)
        if (rc != 0 && rc != DEGRADED) {
            error("production_performance: exit ${rc}")
        }
        degraded = rc == DEGRADED
    }

    stage('Entrenar y promover') {
        def rc = compose('run --rm trainer', true)
        if (rc == CHALLENGER_LOST) {
            if (degraded) {
                unstable('El champion está degradado con las ventas reales y el challenger no lo supera: '
                         + 'revisar el modelo (MLflow, Grafana -> House Price Predictor -> Error real).')
                currentBuild.description = 'champion degradado, challenger perdió'
                notify('degraded_lost', currentBuild.description)
            } else {
                unstable('El challenger perdió contra el champion: sigue en producción la versión actual.')
                currentBuild.description = 'challenger perdió'
                notify('challenger_lost', currentBuild.description)
            }
            return
        }
        if (rc != 0) {
            error("Entrenamiento o puerta de calidad: exit ${rc}")
        }
    }
    if (currentBuild.description?.contains('challenger perdió')) {
        return
    }

    stage('Desplegar') {
        compose('restart house-price-api')
        sh '''
            for _ in $(seq 1 40); do
              [ "$(docker inspect -f '{{.State.Health.Status}}' house-price-api)" = healthy ] && exit 0
              sleep 3
            done
            docker logs --tail 50 house-price-api
            exit 1
        '''
        def expected = sh(script: 'cd /home/ubuntu/house-price-predictor && docker compose run --rm -T trainer '
                                  + 'python scripts/continuous_training.py champion 2>/dev/null | tail -1',
                          returnStdout: true).trim()
        def serving = sh(script: '''docker exec house-price-api python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/model-info'))['model_version'])"''',
                         returnStdout: true).trim()
        if (serving != expected) {
            error("La API sirve la versión ${serving} y @champion es la ${expected}")
        }
        currentBuild.description = "v${serving} en producción" + (degraded ? ' (reemplaza a un champion degradado)' : '')
        notify('promoted', currentBuild.description)
    }
  } catch (e) {
    notify('failed', e.getMessage() ?: e.toString())
    throw e
  }
}
