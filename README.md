*In Russian*
Сегодня 28.10.2025, я выложил данный проект на github.
Идея проекта пришла мне в начале месяца октябрь, когда я изучал материал про нейросети 11 класса. Я настолько не понимал написанное, что начал искать больше информации в интернете. В следствие чего я решил сделать свою нейросеть без использования специализированных модулей.
Для упрощения были использованы слелдующий модули
1) numpy, для упрощения работы с матрицами
2) matplotlib, для графического представления результатов обучения

Описание работы программного кода:
Мы создаем функции для работы кода
1) generate_digit_matrix(digit, size=10) - данная функция создает чистое матричное представление цифр
2) create_dataset(samples_per_digit=200, noise_level=0.2) - данная функция создает набор данных цифр в которые был добавлен шум
3) Подготовливаем данные X_train, y_train, X_val, y_val
4) Инициализация нейросети, а также создание seed
5) Создаем функцию для расчета ReLU(x) - для исправления исчезающего градиента
6) Создаем функцию stable_softmax(z) - для расчета многопеременной логистической — это обобщение логистической функции для многомерного случая
7) Создаем функцию cross_entropy_loss(probs, targets) - расчет кросс-энтропии
8) Создаем функцию accuracy(probs, targets) - расчет точности
9) Создаем функцию train(X_train, y_train, X_val=None, y_val=None) - основная функция в которой проходит обучение
10) Далее визуализируем результаты

P.S.
Спасибо за внимание



*In English"
Today, October 28, 2025, I posted this project on github.
The idea for this project came to me in early October, when I was studying the neural network material for 11th grade. I was so confused by what I'd read that I started searching for more information online. Consequently, I decided to create my own neural network without using specialized modules.
For simplification, the following modules were used
1) numpy, to simplify working with matrices
2) matplotlib, for graphical representation of training results
3) 
Description of the program code:
We create functions for the code to work.
1) generate_digit_matrix(digit, size=10) - this function creates a pure matrix representation of digits.
2) create_dataset(samples_per_digit=200, noise_level=0.2) - this function creates a dataset of digits with added noise.
3) Prepare the data: X_train, y_train, X_val, y_val.
4) Initialize the neural network and create a seed
5) Create a function to calculate ReLU(x) - to correct for the vanishing gradient
6) Create a function stable_softmax(z) - to calculate the multivariable logistic function - this is a generalization of the logistic function for the multivariate case
7) Create a function cross_entropy_loss(probs, targets) - to calculate the cross-entropy
8) Create the function accuracy(probs, targets) - calculate accuracy
9) Create the function train(X_train, y_train, X_val=None, y_val=None) - the main function in which training occurs
10) Next, visualize the results

P.S.
Thank you for your attention.
