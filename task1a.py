import csv
from textdistance import jaccard as jac
from textdistance import levenshtein as lvs
import re
MANU_SIM = 0.35

def normalise_string(text):
    """ Removes punction and decapitalises text """
    text = "".join(t for t in text if t not in ("?", ".", ";", ":", "!", "-", "/", "|","\\", "\"", "\'"))
    return text.lower()

def sim_manufacturer(a_row, b_row):
    """ Returns sim of manufacturer """
    manu = is_same_manufacturer(a_row, b_row)
    global MANU_SIM
    if not manu:  # Not the same manu
        return -5
    
    if manu == 'unknown':  # Missing Manu
        return 0
    
    return MANU_SIM  # Same manu


def is_same_manufacturer(a_row, b_row):
    """ Compares if a_row and b_row has the same manufacturer, returns b's manufacturer if true, empty string if False, 'unknown' if b's manufacturer is missing """
    
    a_name = normalise_string(a_row[1])
    a_desc = normalise_string(a_row[2])
    b_manu = normalise_string(b_row[3])

    if not b_manu:
        return 'unknown'
    
    if (a_name.split()[0] in b_manu or b_manu in  a_name.split() or b_manu in a_desc.split()):
        return b_row[3]
    
    return ''

def sim_name_desc(a_row, b_row):
    """ Returns similarity between names and descriptions of a_row and b_row based  """
    a_name = normalise_string(''.join(a_row[1].split()[1:]))  # Excluding first word which is the manufacturer
#    a_desc = normalise_string(''.join(a_row[2].split()[1:]))
    b_name = normalise_string(''.join(b_row[1].split()[1:]))
#    b_desc = normalise_string(''.join(b_row[2].split()[1:]))

    sim = (jac(a_name, b_name)  + lvs.normalized_similarity(a_name, b_name)) 
    return sim

def sim_numbers_unicode(a_row, b_row, sor_sim, manu_sim):
    """ Checks how similar the unicode and numbers that are contained in names and descriptions """
    global MANU_SIM
    STRONG_SIM_SCORE = 0.8
    INDICATOR_SIM_SCORE = 0.15
    MINUS_IND = -0.25
    sim = 0
    
    SAME_DESIGN_SOR = 0.6
    strong_indicators_a = re.findall(r'[a-z]+\d+[a-z]+|[a-z]+\d+[a-z]+\d+|\d+[a-z]+\d+|\d+\s+x\s+\d+',normalise_string(a_row[1]))
    strong_indicators_a.extend(re.findall(r'[a-z]+\d+[a-z]+|[a-z]+\d+[a-z]+\d+|\d+[a-z]+\d+|\d+\s+x\s+\d+',normalise_string(a_row[2])))

    indicators_a = re.findall(r'\d+[a-z]+|[a-z]+\d+|\d+',normalise_string(a_row[1]))
    indicators_a.extend(re.findall(r'\d+[a-z]+|[a-z]+\d+|\d+',normalise_string(a_row[2])))      
    indicators_a = [x for x in indicators_a if x not in ' '.join(list(strong_indicators_a))]  # remove duplicates

    b_name = normalise_string(b_row[1])
    b_desc = normalise_string(b_row[2])

    
    for ind in indicators_a:
        if len(ind) > 4 and (ind in b_name or ind in b_desc) and ind.count('0')/len(ind) <=0.34:
            sim += MANU_SIM  
    
    for ind in strong_indicators_a:  # check strong indicators
        if ind in b_name or ind in b_desc:
            sim += STRONG_SIM_SCORE
            
    for ind in indicators_a:  # check normal indicators
        if ind in b_name or ind in b_desc:
            sim += INDICATOR_SIM_SCORE
    
    if sim ==0 and (strong_indicators_a or len(indicators_a) > 2):
        return MINUS_IND
    else: 
        return sim

def sim_score(a_row, b_row):
    """ Returns sim of scores """
    LOW_PRICE = 40
    LOW_DIF = 22
    MEDIUM_PRICE = 200
    PERCENT = 0.4
    NEG_SIM = -1
    SIM_SCORE = 0.2
    if a_row[3] and b_row[4]:
        a_price = float("".join(t for t in a_row[3] if t not in ("$", ",", " ")))
        b_price = float("".join(t for t in b_row[4] if t not in ("$", ",", " ")))
        the_max = max(a_price, b_price)  
        the_min = min(a_price, b_price)
        
        if the_max < LOW_PRICE:
            if (the_max - the_min) > LOW_DIF:
                return NEG_SIM
            
        if the_max < MEDIUM_PRICE:
            if (the_max - the_min) > the_min:
                return NEG_SIM
            
        if ((the_max - the_min) / the_max) > PERCENT:
            return NEG_SIM

        return SIM_SCORE
            
    return 0

         
abt_file = open("abt.csv","r", encoding = 'ISO-8859-1')
buy_file = open("buy.csv", "r", encoding = 'ISO-8859-1')
reader_a = list(csv.reader(abt_file))
reader_b = list(csv.reader(buy_file))

output = open("task1a.csv",'w')
writer = csv.writer(output)
writer.writerow([reader_a[0][0], reader_b[0][0]])

for a_row in reader_a[1:]:
    for b_row in reader_b[1:]:
        sor_sim = sim_name_desc(a_row, b_row)
        manu_sim = sim_manufacturer(a_row, b_row)
        sim =  sim_score(a_row, b_row) + manu_sim +(sor_sim) + sim_numbers_unicode(a_row, b_row, sor_sim, manu_sim)
        if sim >= 1.92:  # Positive
            writer.writerow([a_row[0], b_row[0]])
            
abt_file.close()
buy_file.close()
output.close()

