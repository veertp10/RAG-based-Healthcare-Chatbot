pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/veertp10/RAG-based-Healthcare-Chatbot.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    if [ ! -d "venv" ]; then
                        python3 -m venv venv
                    fi
                    . venv/bin/activate
                    pip install -r requirement.txt
                '''
            }
        }

        stage('Run Application') {
            steps {
                sh '''
                    . venv/bin/activate
                    python chatpdf1.py
                '''
            }
        }
    }
}
