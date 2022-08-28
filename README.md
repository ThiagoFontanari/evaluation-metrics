# MODEL PROPOSAL | PROPOSTA DO MODELO

This model aims to demonstrate the application of evaluation metrics on a model, in order to evaluate the quality of its learning. In this example, a model was implemented from a ResNet50 network, using the training weights of the Imagenet base. The proposal is to verify how this model behaves when used to identify defects on train tracks. For training, a database obtained from https://www.kaggle.com/datasets/gpiosenka/railway-track-fault-detection-resized-224-x-224 was used.<br />

Este modelo tem como objetivo demonstrar a aplicação de métricas de avaliação sobre um modelo, afim de avaliar a qualidade do seu aprendizado. Neste exemplo, foi implementado um modelo a partir de uma rede ResNet50, utilizando os pesos de treinamento da base Imagenet. A proposta é verificar como esse modelo se comporta ao ser utilizado para identificar defeitos em trilhos de trem. Para o treinamento, foi utilizada uma base de dados obtida em https://www.kaggle.com/datasets/gpiosenka/railway-track-fault-detection-resized-224-x-224.<br />

After implementing the model, we have the following network structure | Após a implementação do modelo temos a seguinte estrutura de rede:<br />

![](https://github.com/ThiagoFontanari/evaluation-metrics/blob/master/img/summary.png)<br />

Below, we can see the model's loss evolution during training: | Abaixo, pode-se verificar a evolução da perda do modelo durante o treinamento:<br />

![](https://github.com/ThiagoFontanari/evaluation-metrics/blob/master/img/loss.png)<be />

With the trained model, predictions were made on data that did not participate in the training, and an accuracy of 64% was reached in the predictions. | Com o modelo treinado, foram realizadas previsões sobre dados que não participaram do treinamento, e chegou-se à uma acurácia de 64% nas previsões.<br />

![](https://github.com/ThiagoFontanari/evaluation-metrics/blob/master/img/acc.png)<br />

With the generation of the confusion matrix from the forecast data, it is noted that, despite the low accuracy of the model, there was 100% accuracy in the forecast where the identification of defects was expected | Com a geração da matriz de confusão a partir dos dados das previsões, nota-se que, apesar da baixa precisão do modelo, houve 100% de acerto na previsão onde se esperava a identificação dos defeitos:<br />

![](https://github.com/ThiagoFontanari/evaluation-metrics/blob/master/img/matrix.png)<br />

From the matrix, the values of other metrics are obtained | A partir da matriz, obtém-se os valores de outras métricas:<br />
	**Sensibilidade | Sensitivity** = 0.58<br />
	**Especificidade | Specificity** = 1<br />
	**Acurácia | Accuracy** = 0.64<br />
	**Precisão | Precision** = 1<br />
	**F-score** = 0.73<br />
