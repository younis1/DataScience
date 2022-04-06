import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import csv
import json
from numpy import arange
import matplotlib.pyplot as plt

# Crawling method very similar to one in workshop 5

MAX_RUGBY_SCORE = 162  # I searched that from Google, this is needed to know when to differiante a date from score
MIN_RUGBY_SCORE = 0

def check_team_name(article,names):
    """ returns first team mentioned in article, or empty string if none found """
    pattern = r"("
    for name in names:
        pattern += name+"|"
    pattern = pattern[:-1]+ ")"  # To remove last "|"

    # Check if any name was found
    team_names = re.findall(pattern, article)

    # Name found
    if team_names:
        return team_names[0]
    
    # No name found
    return []
        

def refine_scores(all_scores, MAX_SCORE, MIN_SCORE):
    """ changes a list of scores to list of valid scores (within MAX_SCORE and MIN_SCORE) """
    for this_score in all_scores:
        (x,y) = this_score.split(sep='-')

        # If invalid score
        if not (MIN_RUGBY_SCORE <= int(x) <= MAX_RUGBY_SCORE and MIN_RUGBY_SCORE <= int(y) <= MAX_RUGBY_SCORE):
            all_scores.remove(this_score)  

def get_largest_score(all_scores):
    """ returns largest score based on the sum of points, then by the score containing the largest number """

    # Empty score
    if not all_scores:
        return []
    
    sorted_scores = sorted(all_scores, key=lambda x: (int(x.split("-")[0]) + int(x.split("-")[1]), max(int(x.split("-")[0]), int(x.split("-")[1]))), reverse=True)
    return sorted_scores[0]

def get_average_difference(scores):
    """ Returns the average score from a list of scores by adding the difference and dividing it by the number of scores """
    total = 0
    for i in scores:
        total += abs(int(i.split("-")[0]) - int(i.split("-")[1]))
    return total/len(scores)

with open('rugby.json', 'r') as outfile:
    data = json.load(outfile)

# Stores all the names of json file inside the list "names"
names = [x["name"] for x in data["teams"]]

# Dictionaries for number of articles, all largest scores of, avg_score of each team
names_articles_dict = {k:0 for k in names} 
names_scores_dict = {k:[] for k in names}
names_avg_score_dict = {k:0 for k in names}

# Specify the initial page to crawl # From Workshop
base_url = 'http://comp20008-jh.eng.unimelb.edu.au:9889/main/'
seed_item = 'index.html'

seed_url = base_url + seed_item
page = requests.get(seed_url)
soup = BeautifulSoup(page.text, 'html.parser')

visited = {} 
visited[seed_url] = True
pages_visited = 1

# Creates csv for task 1
csv_file_1 = open('task1.csv','w')
writer_1 = csv.writer(csv_file_1)
writer_1.writerow(['url', 'headline'])

# Creates csv for task 2
csv_file_2 = open('task2.csv','w')
writer_2 = csv.writer(csv_file_2)
writer_2.writerow(['url', 'headline', 'team', 'score'])

# Creates csv for task 3
csv_file_3 = open('task3.csv','w')
writer_3 = csv.writer(csv_file_3)
writer_3.writerow(['team','avg_game_difference'])

#Remove index.html  # From Workshop
links = soup.findAll('a')
seed_link = soup.findAll('a', href=re.compile("^index.html"))
to_visit_relative = [l for l in links if l not in seed_link]

# Resolve to absolute urls  # From Workshop
to_visit = []
for link in to_visit_relative:
    to_visit.append(urljoin(seed_url, link['href']))

    
#Find all outbound links on succsesor pages and explore each one
while (to_visit):
       
    # consume the list of urls
    link = to_visit.pop(0)
    page = requests.get(link)

    # scraping code goes here
    soup = BeautifulSoup(page.text, 'html.parser')
    headline = soup.find('h1', class_ = "headline").text
    
    # To get article in a string, we start with headline followed by space, then we add each paragraph followed by a space
    article = headline + " "
    for i in range(len(soup.find_all('p'))):
        article += soup.find_all('p')[i].text + " "

    # Get team name (First name mentioned)
    team_name = check_team_name(article, names)

    # Get scores that are within range ( x-y) format
    all_scores = re.findall(r' \d+-\d+',article)

    # Differentiate scores from dates by removing scores out of range from the list "all_scores"
    refine_scores(all_scores, MAX_RUGBY_SCORE, MIN_RUGBY_SCORE)
    
    # Get largest score based on sum of score, then the score containing the largest number if sum is equal
    largest_score = get_largest_score(all_scores)

    # Write to csv file 1 for task 1
    writer_1.writerow([link, headline])
    
    # Add 1 to names_articles_dict if there is a team_name, regardless if there is a score or not
    if team_name:
        names_articles_dict[team_name] += 1
        
        # If game contains team_name and a valid score, Write to csv file 2 for task 2, append largest_score to names_scores_dict
        if largest_score:
            writer_2.writerow([link, headline, team_name, largest_score])
            names_scores_dict[team_name].append(largest_score)


    
    # Mark the item as visited, i.e., add to visited list, remove from to_visit # From Workshop
    visited[link] = True
    new_links = soup.findAll('a')
    for new_link in new_links :
        new_item = new_link['href']
        new_url = urljoin(link, new_item)
        if new_url not in visited and new_url not in to_visit:
            to_visit.append(new_url)
        
    pages_visited = pages_visited + 1

# Plot for task 4 by sorting, and taking top 5 mentioned teams
sorted_freq_teams = sorted(names_articles_dict, key=lambda x:(-names_articles_dict[x]))
top_5_freq_teams = sorted_freq_teams[:5]
top_5_freq = [names_articles_dict[x] for x in top_5_freq_teams]

# Similar to workshop method, adding title, and axis-labels
plt.bar(arange(len(top_5_freq)), top_5_freq)
plt.xticks(arange(len(top_5_freq_teams)), top_5_freq_teams)
plt.xlabel("Team Name")
plt.ylabel("Number of articles")
plt.title("Task4 Graph")
plt.savefig('task4.png')
plt.close()

# Prepare names_avg_score_dict for task 5
for (this_team, this_score) in names_scores_dict.items():
    writer_3.writerow([this_team, get_average_difference(this_score)])
    names_avg_score_dict[this_team] = get_average_difference(this_score)


# Plot for task 5
average_scores = []
articles_mentioned = []
for this_name in sorted_freq_teams:
    average_scores.append(names_avg_score_dict[this_name])
    articles_mentioned.append(names_articles_dict[this_name])
    
# Similar to workshop method,  adding a legend, title, and axis-labels
plt.bar(arange(len(average_scores))-0.3, average_scores, width=0.3, color = 'g',label='average score')
plt.bar(arange(len(articles_mentioned)),articles_mentioned, width=0.3,color='b',label = 'Number of articles')
plt.xticks(arange(len(names)),names, fontsize = 8)
plt.legend()
plt.xlabel("Team Name")
plt.ylabel("Frequency")
plt.title("Task5 Graph")
plt.savefig("task5.png")
plt.close()

# Close all csv files
csv_file_1.close()
csv_file_2.close()
csv_file_3.close()
