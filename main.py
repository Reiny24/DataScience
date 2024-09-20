import requests, bs4, pandas, numpy, math
import matplotlib.pyplot as plt
from scipy import stats

def parse_site_tesla_sales(url):
    '''
        Функція парсингу сайту історії цін акцій Тесли

        :param url: посилання на вебсайт
        :return dict: результат парсингу у виді словника
    '''
    # Імітація запиту користувача використовуючи headers
    # response = requests.get(url, headers=headers)
    # print(response.status_code)
    # Збереження відповіді сайту у файл за для уникнення блокування 
    # with open("response.txt", "w", encoding="utf-8") as file:
    #     file.write(response.text)
    with open("response.txt", "r", encoding="utf-8") as file:
        webpage_text = file.read()
    
    columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    result_dict = {key: [] for key in columns}
    
    soup = bs4.BeautifulSoup(webpage_text, "html.parser")
    table = soup.find_all('td')

    for i in range(0, len(table), 7):   # Ітеруємо по таблиці групами по 7 (1 рядок)
        for j, column in enumerate(columns):
            result_dict[column].append(table[i + j].text)
    
    # Конвертуємо дані "volume" з рядків у цілі числа
    result_dict['volume'] = [int(vol.replace(',', '')) for vol in result_dict['volume']]
    # print(result_dict)
    return result_dict

def MNK_Stat_char(S0, output=True):
    '''
        Функція обчислення коефіцієнтів тренду та трендової складової 
        з використанням методу найменших квадратів. Вивід регресійної моделі
        
        :param S0: масив вхідних даних
        :return Yout, C: трендова складова та коефіцієнти тренду
    '''
    n_iter = len(S0)
    Yin = numpy.array(S0).reshape(n_iter, 1)     # вектор вхідних значень
    # Розв'язання системи рівнянь FT x F x C = FT x Y
    F = numpy.vstack([numpy.ones(n_iter), numpy.arange(n_iter), numpy.arange(n_iter)**2]).T
    C = numpy.linalg.solve(F.T.dot(F), F.T.dot(Yin))
    Yout = F.dot(C)
    if output: print(f'Регресійна модель:\ny(t) = {C[0,0]} + {C[1,0]}t + {C[2,0]}t^2')
    return [C, Yout]

# Статистичні характеристики вхідної вибірки
def Stat_char_in (SL, text, remove_trend=True, return_scvS=False):
    print(f'\n{text}')
    Yout = MNK_Stat_char(SL)[1]
    n_iter = len(Yout)
    SL0 = numpy.zeros(n)
    for i in range(n):      
        SL0[i] = SL[i].item()
        if remove_trend: SL0[i] -= Yout[i, 0]   # Усунення тренду
    mS = numpy.mean(SL0)    # Матиматичне очікування
    dS = numpy.var(SL0)     # Дисперсія
    scvS = math.sqrt(dS)    # Середньоквадратичне відхилення
    print(f'К-сть елементів вибірки: {n_iter}')
    print(f'Матиматичне очікування: {mS}')
    print(f'Дисперсія: {dS}')
    print(f'Середньоквадратичне відхилення: {scvS}')
    if return_scvS: return scvS

# Статистичні характеристики лінії тренда
def Stat_char_out(SL_in, SL, text):
    print(f'\n{text}')
    Yout = MNK_Stat_char(SL)[1]
    # глобальне лінійне відхилення оцінки - динамічна похибка моделі
    Delta = 0
    for i in range(n):
        Delta = Delta + abs(SL_in[i] - Yout[i, 0])
    print(f'Динамічна похибка моделі = {Delta / (n + 1)}')

# Cтатистичні характеристики екстраполяції
def Stat_characteristics_extrapol (koef, SL, text):
    # Статистичні характеристики вибірки з урахуванням видалення тренду
    scvS = Stat_char_in(SL, text, return_scvS=True)
    # Довірчий інтервал прогнозованих значень за СКВ
    scvS_extrapol = scvS * koef
    print(f'Довірчий інтервал прогнозованих значень за СКВ: {scvS_extrapol}')

# МНК прогнозування
def MNK_Extrapolation (S0, koef):
    C = MNK_Stat_char(S0)[0]
    Yout_Extrapol = numpy.zeros((n + koef, 1))
    for i in range(n + koef):   # поліноміальна крива МНК - прогнозування
        Yout_Extrapol[i, 0] = C[0, 0] + C[1, 0] * i+(C[2, 0] * i * i)
    return Yout_Extrapol

# Виявлення аномальних відхилень за алгоритмом sliding window
def Sliding_Window_AV_Detect_sliding_wind (S0, n_Wind):
    # Параметри циклів
    j_Wind = math.ceil(n - n_Wind) + 1
    S0_Wind = numpy.zeros(n_Wind)
    Midi = numpy.zeros(n)
    # Ковзне вікно
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        # Стат. хар. ковзного вікна
        Midi[l] = numpy.median(S0_Wind)
    # ---- очищена вибірка  -----
    S0_Midi = numpy.zeros(n)
    for j in range(n):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi

# Обчислення коефіцієнту детермінації - оцінювання якості моделі
def r2_score(SL, Yout, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
        denominator_1 = denominator_1 + SL[i]
    denominator_2 =  0
    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2
    R2_score = 1 - (numerator / denominator_2)
    print(f'\n{Text}')
    print('Коефіцієнт детермінації R^2 (ймовірність апроксимації): ', R2_score)

    return R2_score

# Синтез даних
def synthesize_data(coefs, residual_mean, residual_std, total_length):
    '''
    Функція синтезу даних на основі трендової моделі та статистичних характеристик залишків.
    
    :param coefs: Коефіцієнти трендової моделі (C0, C1, C2)
    :param residual_mean: Середнє значення залишків
    :param residual_std: Середньоквадратичне відхилення залишків
    :param total_length: Загальна кількість точок даних
    :return synthetic_data: Синтезовані дані
    '''
    t = numpy.arange(total_length)
    trend = coefs[0, 0] + coefs[1, 0] * t + coefs[2, 0] * t**2
    noise = numpy.random.normal(loc=residual_mean, scale=residual_std, size=total_length)
    synthetic_data = trend + noise
    return synthetic_data

# Функціє зображення графіків
def draw_plots(title, xlabel, ylabel, data_arr=None, bins=0, plots_arr=None):
    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if data_arr is not None:
        plt.plot(data_arr[0], data_arr[1])
    elif bins != 0:
        plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
    elif plots_arr is not None:
        for plot in plots_arr:
            plt.plot(plot[0], label=plot[1], alpha=0.7)
        plt.legend()
    plt.show()

# 
# Завдання 1. Парсинг сайту
# 
url_template = 'https://finance.yahoo.com/quote/TSLA/history/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
dataframe = pandas.DataFrame(data=parse_site_tesla_sales(url_template))

#
#  Завдання 2. Збереження результату парсингу у файл
# 
dataframe.to_csv("tesla_sales_data.csv")

# Зчитування даних з файлу та візуалізація
data = pandas.read_csv('tesla_sales_data.csv')
# Перетворення дати на datetime формат
data['date'] = pandas.to_datetime(data['date'])
real_data = data['open']    # Обираємо стовпець open (відкриті ціни)
draw_plots('Коливання цін акцій Тесла за останній рік', 'Індекс', 'Відкрита ціна у $',
           data_arr=[data['date'], data['open']])

# 
# Завдання 3. Оцінка динаміки тренду 
# 
n = len(real_data)
print(f'К-сть елементів: {n}')
#  Коефіцієнти тренду та трендова складова
coefs, trend_part = MNK_Stat_char(real_data)

# Очищення від аномальних похибок алгоритмом sliding window
n_Wind = 5  # розмір ковзного вікна для виявлення аномальних відхилень
S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind(real_data, n_Wind)
Stat_char_in(S_AV_Detect_sliding_wind, 'Вибірка очищена від аномальних відхилень (АВ) алгоритм sliding_wind')

# 
# МНК згладжування
# 
Yout_SV_AV_Detect_sliding_wind = MNK_Stat_char(S_AV_Detect_sliding_wind, False)[1]
Stat_char_out(real_data, Yout_SV_AV_Detect_sliding_wind, 'MNK згладжена, вибірка очищена від аномальних відхилень алгоритм sliding_wind')
# Оцінювання якості моделі та візуалізація
r2_score(S_AV_Detect_sliding_wind, Yout_SV_AV_Detect_sliding_wind, 'MNK модель згладжування')
draw_plots('MNK згладжування, вибірка очищена від АВ алгоритм sliding_wind', 'Індекс', 'Відкрита ціна у $',
           plots_arr=[(S_AV_Detect_sliding_wind, 'Згладжування'), (Yout_SV_AV_Detect_sliding_wind, 'Тренд')])

# 
# МНК екстраполяція
# 
koef_Extrapol = 0.2  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
koef = math.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки
Yout_SV_AV_Detect_sliding_wind = MNK_Extrapolation(S_AV_Detect_sliding_wind, koef)
# Статистичні характеристики екстраполяції та візуалізація
Stat_characteristics_extrapol(koef, Yout_SV_AV_Detect_sliding_wind, 'MNK прогнозування після очищення від АВ')
draw_plots('MNK прогнозування: Вибірка очищена від АВ алгоритм sliding_wind', 'Індекс', 'Відкрита ціна у $',
           plots_arr=[(S_AV_Detect_sliding_wind, 'Згладжені дані'), (Yout_SV_AV_Detect_sliding_wind, 'Прогнозування')])

# 
# Завдання 4. Статистичні характеристики
# 
Stat_char_in(real_data, 'Статистичні характеристики відкритих цін', False)
Stat_char_in(real_data, 'Статистичні характеристики після усунення тренду')
    
# 
# Завдання 5. Синтезувати та верифікувати модель даних
# 
# Обчислення залишкових похибок (реальні дані - тренд)
residuals = real_data - trend_part.flatten()
# Статистичні характеристики залишків
Stat_char_in(residuals, 'Статистичні характеристики залишків', False)
residual_mean = numpy.mean(residuals)
residual_std = numpy.std(residuals)
# Візуалізація
draw_plots('Гістограма залишкових похибок', 'Залишкові похибки', 'Частота', bins=30)

# Синтез даних
synthetic_open = synthesize_data(coefs, residual_mean, residual_std, n)
# Обчислення статистичних характеристик реальних та синтезованих даних
Stat_char_in(real_data, "Реальні дані", False)
Stat_char_in(synthetic_open, "Синтезовані дані", False)
real_mean = numpy.mean(real_data)
real_std = numpy.std(real_data)
synthetic_mean = numpy.mean(synthetic_open)
synthetic_std = numpy.std(synthetic_open)

# Верифікація синтезованої моделі
print('Верифікація синтезованої моделі:')
# Статистичний тест (t-test) для перевірки різниці середніх
t_stat, p_value = stats.ttest_ind(real_data, synthetic_open)
print(f"T-тест: t-statistic = {t_stat}, p-value = {p_value}")
if p_value > 0.05:
    print("Немає статистично значущої різниці між середніми реальних та синтезованих даних.")
else:
    print("Є статистично значуща різниця між середніми реальних та синтезованих даних.")
# Обчислення достовірності апроксимації R^2
# slope, intercept, r_value, p, std_err = stats.linregress(real_data, synthetic_open)
# print(f"Достовірність апроксимації R^2 (коефіцієнт детермінації): {r_value**2}")
r2_score(numpy.array(real_data).reshape(-1, 1), numpy.array(synthetic_open).reshape(-1, 1), "")

# Візуалізація реальних даних та даних синтезованої моделі
draw_plots('Порівняння реальних та синтезованих даних', 'Індекс', 'Відкрита ціна у $', 
           plots_arr=[(real_data, 'Реальні дані'), (synthetic_open, 'Синтезовані дані')])
