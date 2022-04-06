# Put task1b.py code here
import csv

# Open Files
def normalise_string(text):
    """ Removes punction and decapitalises text """
    text = "".join(t for t in text if t not in ("?", ".", ";", ":", "!", "-", "/", "|","\\", "\"", "\'"))
    return text.lower()

abt_file = open("abt.csv", "r", encoding = 'ISO-8859-1')
buy_file = open("buy.csv", "r", encoding = 'ISO-8859-1')
reader_a = list(csv.reader(abt_file))
reader_b = list(csv.reader(buy_file))

abt_blocks = open('abt_blocks.csv', 'w')
buy_blocks = open('buy_blocks.csv', 'w')

writer_a   = csv.writer(abt_blocks)
writer_b   = csv.writer(buy_blocks)

writer_a.writerow(['block_key', 'product_id'])
writer_b.writerow(['block_key', 'product_id'])


# Blocking on manufacturer
b_manu = []

for b_row in reader_b[1:]:
    try:
        b_manu.append(normalise_string(b_row[3]).split(" ")[0])
        writer_b.writerow([normalise_string(b_row[3]).split(" ")[0], b_row[0]])
    except:
        b_manu.append(normalise_string(b_row[1]).split(" ")[0])
        writer_b.writerow([normalise_string(b_row[1]).split(" ")[0], b_row[0]])
        
for a_row in reader_a[1:]:
    first_word = normalise_string(a_row[1]).split(" ")[0]
    if first_word in b_manu:
        writer_a.writerow([first_word, a_row[0]])


# Close all files
abt_file.close()
buy_file.close()
abt_blocks.close()
buy_blocks.close()
