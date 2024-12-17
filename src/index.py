import numpy as np
import matplotlib.pyplot as plt

# -------------------- تعریف مرزهای هر مجموعه مثلثی --------------------
def create_fuzzy_sets(start, end, count):
    step = (end - start) / (count - 1)
    points = np.linspace(start, end, count)
    fuzzy_sets = [(points[i] - step, points[i], points[i] + step) for i in range(len(points))]
    return fuzzy_sets

# تابع عضویت مثلثی دستی
def triangular_MF(x, a, b, c):
    """
    x: مقدار ورودی
    a, b, c: مرزهای مثلث
    """
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0.0

    # ایجاد مجموعه‌های فازی برای x1, x2 و y
x1_sets = create_fuzzy_sets(-5, 5, 7)
x2_sets = create_fuzzy_sets(-5, 5, 7)
y_sets = create_fuzzy_sets(0, 50, 7)
# -------------------- ساخت قوانین فازی با روش Wang-Mendel --------------------
def create_fuzzy_rules(data_points):
    rules = []



    # محاسبه عضویت‌های ورودی‌ها و خروجی‌ها
    for x1, x2, y in data_points:
        # محاسبه درجه عضویت‌های ورودی‌ها
        x1_fuzzy = [triangular_MF(x1, *fs) for fs in x1_sets]
        x2_fuzzy = [triangular_MF(x2, *fs) for fs in x2_sets]
        
        # محاسبه درجه عضویت برای y
        y_fuzzy = [triangular_MF(y, *fs) for fs in y_sets]

        # انتخاب مجموعه‌هایی که بیشترین درجه عضویت را دارند
        x1_max = np.argmax(x1_fuzzy)
        x2_max = np.argmax(x2_fuzzy)
        y_max = np.argmax(y_fuzzy)

        # پیدا کردن بیشترین درجات عضویت
        x1_max_value = np.max(x1_fuzzy)
        x2_max_value = np.max(x2_fuzzy)
        y_max_value = np.max(y_fuzzy)

        # محاسبه درجه اعتبار (valid_degree) و درجه قانون (rule_degree)
        valid_degree = x1_max_value * x2_max_value * y_max_value
        rule_degree = min(x1_max_value, x2_max_value, y_max_value)

        # دیفازی‌سازی (محاسبه مقدار دیفازی‌شده خروجی y)
        defuzzified_output = defuzzify(y_fuzzy, [fs[1] for fs in y_sets])  # مرکز مجموعه‌های y

        # اضافه کردن قانون به لیست قوانین
        rules.append({
            'x1_set': x1_max,
            'x2_set': x2_max,
            'y_set': y_max,
            'rule_degree': rule_degree,
            'valid_degree': valid_degree,
            'defuzzified_output': defuzzified_output
        })
    
    return rules, x1_sets, x2_sets, y_sets

# -------------------- حذف قوانین تکراری با ورودی مشابه و خروجی متفاوت --------------------
def remove_duplicate_rules_with_highest_degree(rules):
    unique_rules = {}

    for rule in rules:
        x1_index, x2_index = rule['x1_set'], rule['x2_set']
        
        # بررسی وجود قوانین تکراری
        if (x1_index, x2_index) not in unique_rules:
            unique_rules[(x1_index, x2_index)] = rule
        else:
            existing_rule = unique_rules[(x1_index, x2_index)]
            if rule['valid_degree'] > existing_rule['valid_degree']:
                unique_rules[(x1_index, x2_index)] = rule

    # تبدیل دوباره به لیست از قوانین
    final_rules = list(unique_rules.values())
    return final_rules

# -------------------- دیفازی‌سازی با روش میانگین مراکز --------------------
def defuzzify(fuzzy_values, centers):
    """دیفازی‌سازی با روش میانگین مراکز (Center Average)"""
    numerator = sum(mu * center for mu, center in zip(fuzzy_values, centers))  # صورت
    denominator = sum(fuzzy_values)  # مخرج
    return numerator / denominator if denominator != 0 else 0  # جلوگیری از تقسیم بر صفر

# -------------------- اجرای برنامه --------------------
# تولید داده‌ها (x1, x2, y) برای آزمایش
x1_values = np.linspace(-5, 5, 41)
x2_values = np.linspace(-5, 5, 41)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
y_values = x1_grid**2 + x2_grid**2  # y = x1^2 + x2^2

data_points = np.column_stack((x1_grid.ravel(), x2_grid.ravel(), y_values.ravel()))

# ایجاد قوانین فازی
rules, x1_sets, x2_sets, y_sets = create_fuzzy_rules(data_points)

# حذف قوانین تکراری با ورودی مشابه و خروجی متفاوت
unique_rules_with_highest_degree = remove_duplicate_rules_with_highest_degree(rules)
# چاپ قوانین نهایی
print(f"Total unique rules: {len(unique_rules_with_highest_degree)}")
for rule in unique_rules_with_highest_degree:
    print(f"Rule: x1 is A{rule['x1_set']} and x2 is B{rule['x2_set']} then F is C{rule['y_set']} "
          f"with rule degree {rule['rule_degree']} and valid degree {rule['valid_degree']:.3f} "
          f"-> Defuzzified output: {rule['defuzzified_output']:.3f}")


# -------------------- پیش‌بینی خروجی برای داده‌های تست --------------------
def predict_output(x1, x2, rules, x1_sets, x2_sets, y_sets):
    """
    پیش‌بینی خروجی برای یک نقطه تست (x1, x2) با استفاده از قوانین فازی.
    """
    # فازی‌سازی مقادیر x1 و x2
    x1_fuzzy = [triangular_MF(x1, *fs) for fs in x1_sets]
    x2_fuzzy = [triangular_MF(x2, *fs) for fs in x2_sets]

    # جمع‌آوری خروجی‌های قوانین فعال
    numerator = 0.0
    denominator = 0.0

    for rule in rules:
        # عضویت x1 و x2 در مجموعه‌های مربوط به قانون
        mu_x1 = x1_fuzzy[rule['x1_set']]
        mu_x2 = x2_fuzzy[rule['x2_set']]

        # میزان فعال‌سازی قانون
        rule_activation = min(mu_x1, mu_x2)

        # خروجی دیفازی‌شده این قانون
        rule_output = rule['defuzzified_output']

        # ترکیب خروجی‌ها
        numerator += rule_activation * rule_output
        denominator += rule_activation

    # محاسبه خروجی نهایی دیفازی‌شده
    return numerator / denominator if denominator != 0 else 0.0

# -------------------- محاسبه MSE و R^2 برای داده‌های تست --------------------
def evaluate_model(test_data, rules, x1_sets, x2_sets, y_sets):
    y_true = []
    y_pred = []

    for x1, x2, y_actual in test_data:
        # پیش‌بینی خروجی برای هر نقطه تست
        y_estimated = predict_output(x1, x2, rules, x1_sets, x2_sets, y_sets)
        y_true.append(y_actual)
        y_pred.append(y_estimated)

    # محاسبه MSE
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

    # محاسبه R^2
    ss_total = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    ss_residual = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return mse, r2, y_true, y_pred

# -------------------- اجرای بخش ارزیابی مدل --------------------
# تولید داده‌های تست
test_x1_values = np.linspace(-5, 5, 13)
test_x2_values = np.linspace(-5, 5, 13)
test_x1_grid, test_x2_grid = np.meshgrid(test_x1_values, test_x2_values)
test_y_values = test_x1_grid**2 + test_x2_grid**2  # تابع هدف: y = x1^2 + x2^2
test_data = np.column_stack((test_x1_grid.ravel(), test_x2_grid.ravel(), test_y_values.ravel()))

# ارزیابی مدل
mse, r2, y_true, y_pred = evaluate_model(test_data, unique_rules_with_highest_degree, x1_sets, x2_sets, y_sets)

# نمایش نتایج
print(f"Test MSE: {mse:.4f}")
print(f"Test R^2: {r2:.4f}")


# محاسبه MSE به‌صورت صدبار
mse_list = []
for _ in range(100):
    mse, r2, y_true, y_pred = evaluate_model(test_data, unique_rules_with_highest_degree, x1_sets, x2_sets, y_sets)
    mse_list.append(mse)

average_mse = np.mean(mse_list)

# نمایش میانگین MSE
print(f"Average MSE over 100 runs: {average_mse:.4f}")


fig = plt.figure(figsize=(12, 6))

# نمودار Train
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(data_points[:, 0], data_points[:, 1], data_points[:, 2], cmap='viridis')
ax1.set_title('Train Data: F(x1, x2) = x1^2 + x2^2')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('F(x1, x2)')

# نمودار Test
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(test_data[:, 0], test_data[:, 1], y_pred , color='r', label='Predicted')
ax2.scatter(test_data[:, 0], test_data[:, 1], y_true, color='g', label='True')
ax2.set_title('Test Data: Predicted vs True Outputs')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('F(x1, x2)')
ax2.legend()

plt.tight_layout()
plt.show()