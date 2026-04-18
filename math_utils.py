def find_max_number(n1, n2, n3):
    if n1 > n2 :
      if n1 > n3 :
        return n1 
      else :
        return n3
    else :
      if n2 > n3 :
        return n2
      else :
        return n3

def find_mean(num1, num2, num3):
    mean = (num1 + num2 + num3)/3
    return mean

def find_mean_std(num1, num2, num3):
    mean = find_mean(num1, num2, num3)
    std = (((num1 - mean)**2 + (num2 -mean)**2 +(num3 - mean)**2)/3)**0.5
    return mean, std


# --- שלב 1: ייבוא המודולים (ארגז הכלים) ---
import random        # מודול ליצירת מספרים אקראיים (פשוטים)
import numpy as np   # מודול לחישובים מתמטיים מתקדמים (שורש, קבוע פאי)
import matplotlib.pyplot as plt  # המודול שאחראי על הציור והגרפים

# --- שלב 2: הגדרת הפונקציה לחישוב פאי ---
def estimate_pi(n_sample): # פונקציה שמקבלת n_sample: כמות ה"נקודות" שנטיל בניסוי
    count_inside_circle = 0 # משתנה שסופר כמה נקודות פגעו בתוך העיגול
    pi_values = []          # רשימה שתשמור את הערך של פאי אחרי כל הטלה (כדי לראות התכנסות)

    for i in range(n_sample):
        x = random.random() # מגריל מספר בין 0 ל-1 (מיקום ב-x)
        y = random.random() # מגריל מספר בין 0 ל-1 (מיקום ב-y)
        
        radius = 0.5        # חצי מהריבוע (המרכז נמצא ב-0.5, 0.5)
        # חישוב מרחק מהמרכז לפי משפט פיתגורס: $d = \sqrt{(x-0.5)^2 + (y-0.5)^2}$
        distance_to_center = ((x-radius)**2 + (y-radius)**2)**0.5
        
        if distance_to_center <= radius: # אם המרחק קטן מהרדיוס - הנקודה בתוך העיגול
            count_inside_circle += 1
        
        # חישוב פאי לפי היחס בין שטחים (שטח עיגול חלקי שטח ריבוע)
        pi = 4 * count_inside_circle / (i + 1) # i+1 הוא מספר הנקודות שהוטלו עד עכשיו
        pi_values.append(pi) # מוסיף את הניחוש הנוכחי לרשימה
        
    return pi_values # מחזירה את כל רשימת הניחושים לאורך הזמן

# --- שלב 3: הרצת הניסוי 10 פעמים ---
all_last_pi = [] # רשימה לשמירת הניחוש הסופי של כל הרצה (מתוך ה-10)
n_sample = 100000 # מספר הטלות הנקודות בכל סימולציה

for _ in range(10): # לולאה שחוזרת על כל הניסוי 10 פעמים (כדי לראות שזה אקראי)
    pi_values = estimate_pi(n_sample) # קריאה לפונקציה שיצרנו למעלה
    all_last_pi.append(pi_values[-1]) # לוקח את האיבר האחרון (הכי מדויק) ושומר אותו
    plt.plot(pi_values) # מצייר קו של התקדמות הניחוש עבור ההרצה הזו

# --- שלב 4: הוספת "גבולות הטעות" (חוק המספרים הגדולים) ---
# אלו הקווים השחורים המקווקווים שמראים איפה הניחוש "אמור" להיות מבחינה סטטיסטית
large_n_rule = []
for i in range(n_sample):
    # חישוב הגבול העליון: $\pi + \frac{2}{\sqrt{i+1}}$
    large_n_rule.append(2/np.sqrt(i+1) + np.pi)
plt.plot(large_n_rule, '--', c='black', linewidth=5) # ציור הגבול העליון

large_n_rule = [] # איפוס הרשימה לחישוב הגבול התחתון
for i in range(n_sample):
    # חישוב הגבול התחתון: $\pi - \frac{2}{\sqrt{i+1}}$
    large_n_rule.append(-2/np.sqrt(i+1) + np.pi)
plt.plot(large_n_rule, '--', c='black', linewidth=5) # ציור הגבול התחתון

# --- שלב 5: עיצוב הגרף ---
plt.ylim(np.pi - 0.5, np.pi + 0.5) # הגבלת ציר ה-Y כדי שנתמקד באזור של פאי (3.14)
plt.xlim(10, n_sample)             # התחלת הציר מ-10 נקודות (בהתחלה הרעש גדול מדי)
plt.xscale('log')                  # שימוש בסקלה לוגריתמית בציר ה-X (כדי לראות את ההתחלה טוב יותר)

# --- שלב 1: ייבוא המודולים (הכלים שאנחנו צריכים) ---

import pandas as pd 
# קריאה למודול Pandas ומתן כינוי pd. משמש לניהול טבלאות נתונים (DataFrames).

import seaborn as sns 
# קריאה למודול Seaborn ומתן כינוי sns. משמש לגרפים ולטעינת מאגרי מידע מוכנים.

from sklearn.preprocessing import MinMaxScaler 
# מתוך ספריית sklearn, אנחנו מייבאים רק את הפונקציה (מחלקה) MinMaxScaler שאחראית על נורמליזציה.

from sklearn.neighbors import KNeighborsClassifier 
# מתוך ספריית sklearn, מייבאים את האלגוריתם KNeighborsClassifier (סיווג לפי השכנים הקרובים).

# --- שלב 2: טעינה וניקוי הנתונים ---

penguins = sns.load_dataset("penguins") 
# שימוש בפונקציה load_dataset של Seaborn כדי להוריד את טבלת הפינגווינים לתוך המשתנה penguins.

penguins = penguins.dropna() 
# הפעלת הפונקציה dropna() על הטבלה. היא מוחקת כל שורה שיש בה ערך חסר (NaN) כדי שהחישוב לא ישתבש.

# --- שלב 3: חלוקה לתכונות (X) ותשובה (y) ---

X = penguins[['bill length mm', 'body mass g']] 
# בחירת שתי עמודות ספציפיות כ"תכונות" (Features). אלו המשתנים שיעזרו לנו לנחש את סוג הפינגווין.

y = penguins['species'] 
# בחירת עמודת ה"מטרה" (Target). זה מה שאנחנו רוצים שהמודל ילמד לנחש.

# --- שלב 4: הכנת הנתונים (נורמליזציה) ---

scaler = MinMaxScaler() 
# יצירת אובייקט (מופע) של MinMaxScaler. זה "המכשיר" שיבצע את הכיווץ של המספרים.

X_scaled = scaler.fit_transform(X) 
# הפונקציה fit_transform עושה שני דברים:
# 1. fit: לומדת מה המינימום והמקסימום של הנתונים ב-X.
# 2. transform: מחשבת את הנורמליזציה לכל המספרים לפי הנוסחה: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$.



# --- שלב 5: בנייה ואימון המודל ---

knn = KNeighborsClassifier(n_neighbors=5) 
# יצירת המודל. הפרמטר n_neighbors=5 קובע שהמודל יחליט לפי 5 השכנים הכי קרובים.

knn.fit(X_scaled, y) 
# הפונקציה fit "מאמנת" את המודל. היא מחברת בין הנתונים המכווצים (X_scaled) לבין הסוגים (y).



# --- שלב 6: ניבוי על פינגווין חדש ---

new_penguin = pd.DataFrame({'bill length mm': [51], 'body mass g': [4600]}) 
# יצירת טבלה קטנה (DataFrame) עם נתונים של פינגווין חדש שמעולם לא ראינו.

new_penguin_scaled = scaler.transform(new_penguin) 
# שימוש בפונקציה transform. אנחנו מכווצים את הפינגווין החדש לפי אותם חוקים שלמדנו על הפינגווינים הישנים.

y_pred = knn.predict(new_penguin_scaled) 
# הפונקציה predict מבצעת את הניבוי. המודל בודק מי 5 השכנים הכי קרובים לפינגווין החדש ומוציא תשובה.

print(y_pred) 
# הדפסת התוצאה (למשל: 'Adelie' או 'Gentoo').


K-Nearest Neighbors (KNN)
סיכום: מודל סיווג שפועל לפי עיקרון ה"חברים": הוא בודק מי השכנים הכי
קרובים לנתון החדש וקובע את הסוג שלו לפי הרוב. דורש נורמליזציה כי הוא מחשב מרחקים.
import pandas as pd                             # לניהול טבלאות
import seaborn as sns                          # לטעינת נתוני הפינגווינים
from sklearn.preprocessing import MinMaxScaler   # מודול לכיול (נורמליזציה) של מספרים
from sklearn.neighbors import KNeighborsClassifier # האלגוריתם עצמו (סיווג לפי שכנים)
# הכנת הנתונים
df = sns.load_dataset("penguins").dropna()     # טעינת הנתונים ומחיקת שורות ריקות
X = df[['bill_length_mm', 'body_mass_g']]      # המשתנים המסבירים (הקלט של המודל)
y = df['species']                              # משתנה המטרה (מה שאנחנו רוצים לנבא)
# כיול (Scaling) - קריטי ב-KNN!
scaler = MinMaxScaler()                        # יצירת ה"מכייל" (הופך הכל לטווח 0-1)
X_scaled = scaler.fit_transform(X)             # למידה וביצוע הכיול על הנתונים
# בניית המודל
knn = KNeighborsClassifier(n_neighbors=5)      # יצירת המודל. פרמטר: n_neighbors (כמות שכנים לבדיקה)
knn.fit(X_scaled, y)                           # אימון: המודל לומד את המיקום של כל פינגווין
# ניבוי
new = [[50, 4500]]                             # נתונים של פינגווין חדש (אורך מקור, משקל)
new_scaled = scaler.transform(new)             # חובה! לכייל את הנתון החדש באותה צורה
pred = knn.predict(new_scaled)                 # הניבוי הסופי: איזה סוג הפינגווין?

Decision Tree (עץ החלטה)
סיכום: מודל לוגי שבונה מעין "תרשים זרימה" של שאלות כן/לא. הוא מאוד קריא 
ואינטואיטיבי (כמו פרוטוקול אבחון רפואי) ולא דורש נורמליזציה של הנתונים.
from sklearn.tree import DecisionTreeClassifier # המודול לבניית עצי החלטה לסיווג
X = df[['bill_length_mm', 'bill_depth_mm']]    # תכונות הקלט
y = df['species']                              # התשובה שאנחנו מחפשים
# בניית המודל
dt = DecisionTreeClassifier(max_depth=3)       # יצירת העץ. פרמטר: max_depth (עומק העץ/כמות שאלות)
dt.fit(X, y)                                   # אימון: העץ בונה את השאלות הכי חכמות להפרדה
# ניבוי
new_p = pd.DataFrame({'bill_length_mm':[40], 'bill_depth_mm':[15]}) # יצירת נתון חדש בטבלה
y_pred = dt.predict(new_p)                     # הרצת הנתון החדש בתוך ה"שאלות" של העץ

Linear Regression (רגרסיה ליניארית)
סיכום: מודל לחיזוי מספרים רציפים (לא קבוצות). הוא מחפש את הקו 
הישר שמתאר הכי טוב את הקשר בין משתנה אחד (או יותר) לתוצאה מספרית.
from sklearn.linear_model import LinearRegression # המודול לחישוב קווים ישרים וקשרים ליניאריים
טעינה וניקוי
X = df[['flipper_length_mm']]                    # קלט: אורך סנפיר (חייב להיות דו-ממדי [[]])
y = df['body_mass_g']                            # פלט: משקל גוף (המספר שאנחנו רוצים לנבא)
# בניית המודל
lr = LinearRegression()                          # יצירת המודל (ללא פרמטרים מיוחדים בד"כ)
lr.fit(X, y)                                     # אימון: המחשב מוצא את משוואת הקו y = ax + b
# ניבוי
new_data = pd.DataFrame({'flipper_length_mm': [210]}) # נתון של פינגווין עם סנפיר באורך 210
weight_pred = lr.predict(new_data)               # חישוב המשקל הצפוי לפי משוואת הקו

K-Means (אשכולות)
סיכום: למידה לא מונחית. המודל לא יודע את התשובות מראש. 
הוא פשוט מקבל ערימת נתונים ומחלק אותה לקבוצות (אשכולות) לפי דמיון ומרחק גיאומטרי.
from sklearn.cluster import KMeans             # המודול לחלוקת נתונים לקבוצות (אשכולות)
X = df[['bill_length_mm', 'bill_depth_mm']]    # אנחנו נותנים רק תכונות, בלי התשובה (y)!
# בניית המודל
km = KMeans(n_clusters=3)                      # יצירת המודל. פרמטר חובה: n_clusters (כמה קבוצות ליצור)
clusters = km.fit_predict(X)                   # פעולה כפולה: גם לומד את המרכזים וגם משייך כל שורה לקבוצה
# הוספת התוצאה לטבלה המקורית
df['cluster_id'] = clusters                    # יצירת עמודה חדשה עם מספרי הקבוצות (0, 1, 2)






