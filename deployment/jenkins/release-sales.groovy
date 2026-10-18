// Fuente de datos simulada (job house-price-release-sales): publica el mes
// siguiente de ventas de King County en el dataset vivo y dispara el
// entrenamiento continuo. Ver scripts/release_sales.py.
properties([
    disableConcurrentBuilds(),
    buildDiscarder(logRotator(numToKeepStr: '30')),
    parameters([
        string(name: 'INIT_UNTIL', defaultValue: '',
               description: 'Vacío: publica el mes siguiente. AAAA-MM: reinicia el dataset vivo con las ventas hasta ese mes.'),
    ]),
])

timestamps {
    stage('Publicar ventas') {
        def init = params.INIT_UNTIL?.trim()
        if (init && !init.matches('\\d{4}-\\d{2}')) {
            error("INIT_UNTIL debe ser AAAA-MM: ${init}")
        }
        def out = sh(script: 'cd /home/ubuntu/house-price-predictor && docker compose run --rm -T trainer '
                             + "python scripts/release_sales.py ${init ? '--init ' + init : ''}",
                     returnStdout: true).trim()
        echo out
        currentBuild.description = out.split('\n')[-1]
    }

    stage('Entrenamiento continuo') {
        build job: 'house-price-ct', wait: false
    }
}
