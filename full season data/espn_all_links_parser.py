from copy import deepcopy

from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import csv
import re
import enum
from unidecode import unidecode
import re

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('capabilities={"acceptInsecureCerts": True}')

START_DATE = datetime(2023, 10, 24)
END_DATE = datetime(2024, 4, 8)
LINKS_TO_EXCLUDE = ["https://www.espn.com/nba/game/_/gameId/401607495/pacers-lakers"]


def open_website_with_retry(url, driver, retries=3, delay=3):
    attempt = 0

    while attempt < retries:
        try:
            driver.get(url)

            # Check if the page loaded successfully by checking the title or page content
            if "502 Bad Gateway" not in driver.page_source:
                print("Page loaded successfully!")
                break
            else:
                print(f"Attempt {attempt + 1}: 502 Bad Gateway, retrying...")
        except WebDriverException as e:
            print(f"Attempt {attempt + 1} failed with exception: {e}")

        # Wait for a bit before retrying
        time.sleep(delay)
        attempt += 1

    if attempt == retries:
        print("Failed to load the website after multiple attempts.")
    else:
        print("Website loaded successfully.")

    return driver








def main():
    page_url = "https://www.espn.com/nba/schedule/_/date/"
    driver = webdriver.Chrome(options=chrome_options)
    all_links = []
    current_date = START_DATE
    while current_date <= END_DATE:
        print("current date: " + current_date.strftime("%Y%m%d"))
        weekly_url = page_url + current_date.strftime("%Y%m%d")  # Format as YYYY-MM-DD
        current_date += timedelta(days=7)  # Increment by 1 day
        open_website_with_retry(weekly_url, driver)

        elements = driver.find_elements(By.CLASS_NAME, "AnchorLink")
        week_links = []
        for element in elements:
            game_status = element.text
            # Check if the game was postponed
            if "Postponed" in game_status:
                continue
            href = element.get_attribute("href")
            if href and href not in LINKS_TO_EXCLUDE:
                week_links.append(href)
        all_links = all_links + [link for link in week_links if ("gameId" in link and (link not in all_links) and ("west-east" not in link))]
    if current_date > END_DATE:
        print("current date: " + END_DATE.strftime("%Y%m%d"))
        weekly_url = page_url + END_DATE.strftime("%Y%m%d")  # Format as YYYY-MM-DD
        open_website_with_retry(weekly_url, driver)

        elements = driver.find_elements(By.CLASS_NAME, "AnchorLink")
        week_links = []
        for element in elements:
            game_status = element.text
            # Check if the game was postponed
            if "Postponed" in game_status:
                continue
            href = element.get_attribute("href")
            if href and href not in LINKS_TO_EXCLUDE:
                week_links.append(href)
        all_links = all_links + [link for link in week_links if ("gameId" in link and (link not in all_links) and ("west-east" not in link))]
    print("Number of links:" + str(len(all_links)))
    all_playbyplay = []
    for link in all_links:
        playbyplay_list = link.split('/')[:-1]
        playbyplay_list[4] = "playbyplay"
        playbyplay = "/".join(playbyplay_list)
        all_playbyplay.append(playbyplay)
    with open('rs_espn_links.txt', mode='w', newline='') as file:
        for link in all_playbyplay:
            file.write(link + "\n")  # Add a newline after each string
    print(f"links saved")
    print("OK :)")
    driver.quit()



if __name__ == '__main__':
    main()
    # List of NBA team identifiers
    '''
    nba_teams = [
        "hawks", "celtics", "nets", "hornets", "bulls", "cavaliers", "mavericks", "nuggets",
        "pistons", "warriors", "rockets", "pacers", "clippers", "lakers", "grizzlies",
        "heat", "bucks", "timberwolves", "pelicans", "knicks", "thunder", "magic",
        "76ers", "suns", "trail-blazers", "kings", "spurs", "raptors", "jazz", "wizards"
    ]

    # Dictionary to store counts for each team
    team_counts = {team: 0 for team in nba_teams}

    # Read the text file
    with open("/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/rs_espn_links.txt", "r") as file:
        links = file.readlines()

    # Count occurrences of each team in the links
    for link in links:
        for team in nba_teams:
            # Use regex to match whole words (case-insensitive)
            if re.search(rf"\b{team}\b", link, re.IGNORECASE):
                team_counts[team] += 1

    # Print the results
    for team, count in team_counts.items():
        print(f"{team.capitalize()}: {count}")
    '''

