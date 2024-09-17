from copy import deepcopy

from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import csv
import re
import enum

chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('capabilities={"acceptInsecureCerts": True}')

class Plays(enum.Enum):
    MAKES_THREE_POINT_JUMPER = 0
    MISSES_THREE_POINT_JUMPER = 100
    MAKES_TWO_POINT = 200
    MISSES_TWO_POINT = 300
    MAKES_PULLUP = 400
    MISSES_PULLUP = 500
    BLOCK = 600
    DEFENSIVE_REBOUND = 700
    TURNOVER = 800
    SHOOTING_FOUL = 900
    MAKES_FREE_THROW_ONE_OF_ONE = 1000
    MISSES_FREE_THROW_ONE_OF_ONE = 1100
    MAKES_FREE_THROW_ONE_OF_TWO = 1200
    MAKES_FREE_THROW_TWO_OF_TWO = 1300
    MAKES_FREE_THROW_ONE_OF_THREE = 1400
    MAKES_FREE_THROW_TWO_OF_THREE = 1500
    MAKES_FREE_THROW_THREE_OF_THREE = 1600
    MISSES_FREE_THROW_ONE_OF_TWO = 1700
    MISSES_FREE_THROW_TWO_OF_TWO = 1800
    MISSES_FREE_THROW_ONE_OF_THREE = 1900
    MISSES_FREE_THROW_TWO_OF_THREE = 2000
    MISSES_FREE_THROW_THREE_OF_THREE = 2100
    OFFENSIVE_REBOUND = 2200
    MAKES_DRIVING_LAYUP = 2300
    MISSES_DRIVING_LAYUP = 2400
    MAKES_LAYUP = 2500
    MISSES_LAYUP = 2600
    BAD_PASS = 2700
    MAKES_DRIVING_FLOATING_JUMPSHOT = 2800
    MISSES_DRIVING_FLOATING_JUMPSHOT = 2900
    MAKES_THREE_POINT_PULLUP = 3000
    MISSES_THREE_POINT_PULLUP = 3100
    PERSONAL_FOUL = 3200
    MAKES_DRIVING_DUNK = 3300
    MISSES_DRIVING_DUNK = 3400
    MAKES_ALLEY_OOP_DUNK_SHOT = 3500
    MISSES_ALLEY_OOP_DUNK_SHOT = 3600
    MAKES_RUNNING_PULLUP_JUMPSHOT = 3700
    MISSES_RUNNING_PULLUP_JUMPSHOT = 3800
    MAKES_STEPBACK_JUMPSHOT = 3900
    MISSES_STEPBACK_JUMPSHOT = 4000
    MAKES_TIP_SHOT = 4100
    MISSES_TIP_SHOT = 4200
    MAKES_ALLEY_OOPS_LAYUP = 4300
    MISSES_ALLEY_OOPS_LAYUP = 4400
    OFFENSIVE_FOUL = 4500
    LOOSE_BALL_FOUL = 4600
    MAKES_DUNK = 4700
    MISSES_DUNK = 4800
    TRAVELING = 4900
    MAKES_BANK_SHOT = 5000
    MAKES_HOOK_SHOT = 5100
    MISSES_HOOK_SHOT = 5200
    KICKED_BALL_VIOLATION = 5300
    OFFENSIVE_CHARGE = 5400
    VIOLATION = 5500
    MAKES_FINGER_ROLL_LAYUP = 5600
    MISSES_FINGER_ROLL_LAYUP = 5700
    PERSONAL_TAKE_FOUL = 5800
    TRANSITION_TAKE_FOUL = 5900
    DEFENSIVE_THREE_SECONDS = 6000
    MAKES_TECHNICAL_FREE_THROW = 6100
    MISSES_TECHNICAL_FREE_THROW = 6200
    HANGING_TECHFOUL = 6300
    TECHNICAL_FOUL = 6400
    MISSES_BANK_SHOT = 6500
    FLAGRANT_FOUL_1 = 6600
    MAKES_FREE_THROW_FLAGRANT_1_OF_2 = 6700
    MAKES_FREE_THROW_FLAGRANT_2_OF_2 = 6800
    MISSES_FREE_THROW_FLAGRANT_1_OF_2 = 6900
    MISSES_FREE_THROW_FLAGRANT_2_OF_2 = 7000
    MAKES_FREE_THROW_FLAGRANT_1_OF_3 = 7100
    MAKES_FREE_THROW_FLAGRANT_2_OF_3 = 7200
    MAKES_FREE_THROW_FLAGRANT_3_OF_3 = 7300
    MISSES_FREE_THROW_FLAGRANT_1_OF_3 = 7400
    MISSES_FREE_THROW_FLAGRANT_2_OF_3 = 7500
    MISSES_FREE_THROW_FLAGRANT_3_OF_3 = 7600
    UNSPECIFIED_FOUL = 7700
    EJECTED = 7800
    MAKES_RUNNING_JUMPER = 7900
    MISSES_RUNNING_JUMPER = 8000
    DEFENSIVE_TEAM_REBOUND = 8100
    TEAM_REBOUND = 8200
    LOST_BALL = 8300

REBOUND = 0
ASSIST = 1
STEAL = 2
BLOCK = 3
POINT = 4
FOUL = 5
TURNOVER = 6
FOUL_AND_TURNOVER = 7
BOTH_TEAM_FOUL = 8
TEAM_FOUL = 9


HOME = 0
AWAY = 1


DIST_DEFAULT_FT = 0
DIST_WITHIN_4_FT = 1
DIST_5_TO_14 = 2
DIST_15_26 = 3
DIST_27_30 = 4
DIST_30_PLUS = 5



team_fouls = [0,0]
team_turnovers = [0,0]
#rebounds, assists, steals, blocks, points, fouls
box_score = {}

#returns player that assisted, or "None" if unassisted
def removeAssister(play_info):

    assist_pattern = re.compile(rf'\((.*?)\s+assists?\)')
    assist_match = assist_pattern.findall(play_info)
    if not assist_match:
        return ["None"]
    else:
        return assist_match

def removeStealer(play_info):
    steal_pattern = re.compile(rf'\((.*?)\s+steals?\)')
    steal_match = steal_pattern.findall(play_info)
    if not steal_match:
        return ["None"]
    else:
        return steal_match

def name_corrector(player_name, home_roster, away_roster):
    jr_ify = player_name + " Jr."
    if jr_ify in home_roster or jr_ify in away_roster:
        return jr_ify
    if player_name == 'Jeenathan Williams':
        return 'Nate Williams'
    if player_name == "Amare Stoudemire":
        return "Amar'e Stoudemire"
    if player_name == "O.G. Anunoby":
        return "OG Anunoby"
    if player_name == "Bill Walker":
        return "Henry Walker"
    if player_name == "GG Jackson":
        return "GG Jackson II"

    return player_name


#returns numerical value of play. FORMAT: Regex, STAT, VALUE
def playID(play_info, player_name, team):
    global box_score
    global team_fouls
    global team_turnovers
    makes_three_point_jumper = [r"makes (\d+)-foot three point jumper|makes (\d+)-foot three point shot|makes (\d+)-foot three pointer|makes three point jumper|makes three point shot|makes three pointer", POINT, 3]
    misses_three_point_jumper = [r"misses (\d+)-foot three point jumper|misses (\d+)-foot three point shot|misses (\d+)-foot three pointer|misses three point jumper|misses three point shot|misses three pointer", POINT, 0]
    makes_two_point = [r"makes (\d+)-foot two point shot|makes (\d+)-foot jumper|makes two point shot|makes jumper", POINT, 2]
    misses_two_point = [r"misses (\d+)-foot two point shot|misses (\d+)-foot jumper|misses two point shot|misses jumper", POINT, 0]
    makes_pullup = [r"makes (\d+)-foot pullup jump shot|makes pullup jump shot", POINT, 2]
    misses_pullup = [r"misses (\d+)-foot pullup jump shot|misses pullup jump shot", POINT, 0]
    blocks= [r"^blocks", BLOCK, 1]
    defensive_rebound = [r"defensive rebound", REBOUND, 1]
    turnover = [r"\bturnover\b", TURNOVER, 1]
    shooting_foul =  [r"shooting foul", FOUL, 1]
    makes_free_throw_one_of_one = [r"makes free throw 1 of 1", POINT, 1]
    misses_free_throw_one_of_one = [r"misses free throw 1 of 1", POINT, 0]
    makes_free_throw_one_of_two = [r"makes free throw 1 of 2", POINT, 1]
    makes_free_throw_two_of_two = [r"makes free throw 2 of 2", POINT, 1]
    makes_free_throw_one_of_three = [r"makes free throw 1 of 3", POINT, 1]
    makes_free_throw_two_of_three = [r"makes free throw 2 of 3", POINT, 1]
    makes_free_throw_three_of_three = [r"makes free throw 3 of 3", POINT, 1]
    misses_free_throw_one_of_two = [r"misses free throw 1 of 2", POINT, 0]
    misses_free_throw_two_of_two = [r"misses free throw 2 of 2", POINT, 0]
    misses_free_throw_one_of_three = [r"misses free throw 1 of 3", POINT, 0]
    misses_free_throw_two_of_three = [r"misses free throw 2 of 3", POINT, 0]
    misses_free_throw_three_of_three = [r"misses free throw 3 of 3", POINT, 0]
    offensive_rebound = [r"offensive rebound", REBOUND, 1]
    makes_driving_layup = [r"makes driving layup", POINT, 2]
    misses_driving_layup = [r"misses driving layup", POINT, 0]
    makes_layup = [r"makes layup|makes (\d+)-foot layup", POINT, 2]
    misses_layup = [r"misses layup|misses (\d+)-foot layup", POINT, 0]
    bad_pass = [r"bad pass", TURNOVER, 1]
    makes_driving_floating_jumpshot = [r"makes (\d+)-foot driving floating jump shot|makes driving floating jump shot", POINT, 2]
    misses_driving_floating_jumpshot = [r"misses (\d+)-foot driving floating jump shot|misses driving floating jump shot", POINT, 0]
    makes_three_point_pullup = [r"makes (\d+)-foot three point pullup jump shot|makes three point pullup jump shot", POINT, 3]
    misses_three_point_pullup = [r"misses (\d+)-foot three point pullup jump shot|misses three point pullup jump shot", POINT, 0]
    personal_foul = [r"personal foul", FOUL, 1]
    makes_driving_dunk = [r"makes (\d+)-foot driving dunk|makes driving dunk", POINT, 2]
    misses_driving_dunk = [r"misses (\d+)-foot driving dunk|misses driving dunk", POINT, 0]
    makes_alley_oop_dunk_shot = [r"makes (\d+)-foot alley oop dunk shot|makes alley oop dunk shot", POINT, 2]
    misses_alley_oop_dunk_shot = [r"makes (\d+)-foot alley oop dunk shot|misses alley oop dunk shot", POINT, 0]
    makes_running_pullup_jumpshot = [r"makes (\d+)-foot running pullup jump shot|makes running pullup jump shot", POINT, 2]
    misses_running_pullup_jumpshot = [r"misses (\d+)-foot running pullup jump shot|misses running pullup jump shot", POINT, 0]
    makes_stepback_jumpshot = [r"makes (\d+)-foot step back jumpshot|makes step back jumpshot", POINT, 2]
    misses_stepback_jumpshot = [r"misses (\d+)-foot step back jumpshot|misses step back jumpshot", POINT, 0]
    makes_tip_shot = [r"makes (\d+)-foot tip shot|makes tip shot", POINT, 2]
    misses_tip_shot = [r"misses (\d+)-foot tip shot|misses tip shot", POINT, 0]
    makes_alley_oop_layup = [r"makes alley oop layup", POINT, 2]
    misses_alley_oop_layup = [r"misses alley oop layup", POINT, 0]
    offensive_foul = [r"offensive foul", FOUL, 1]
    loose_ball_foul = [r"loose ball foul", FOUL, 1]
    makes_dunk = [r"makes (\d+)-foot dunk|makes dunk|makes slam dunk", POINT, 2]
    misses_dunk = [r"misses (\d+)-foot dunk|misses dunk|misses slam dunk", POINT, 0]
    traveling = [r"traveling", TURNOVER, 1]
    makes_bank_shot = [r"makes (\d+)-foot jump bank shot|makes jump bank shot", POINT, 2]
    makes_hook_shot = [r"makes (\d+)-foot hook shot|makes hook shot", POINT, 2]
    misses_hook_shot = [r"misses (\d+)-foot hook shot|misses hook shot", POINT, 0]
    kicked_ball_violation = [r"kicked ball violation", TURNOVER, 1]
    offensive_charge = [r"offensive charge", FOUL_AND_TURNOVER, 1]
    violation = [r"violation", TURNOVER, 1]
    makes_finger_roll_layup = [r"makes finger roll layup", POINT, 2]
    misses_finger_roll_layup = [r"makes finger roll layup", POINT, 0]
    personal_take_foul = [r"personal take foul", FOUL, 1]
    transition_take_foul = [r"transition take foul", FOUL, 1]
    defensive_three_seconds = [r"defensive 3-seconds", TEAM_FOUL, 1]
    makes_technical_free_throw = [r"makes technical free throw", POINT, 1]
    misses_technical_free_throw = [r"misses technical free throw", POINT, 0]
    hanging_techfoul = [r"hanging techfoul", FOUL, 1]
    technical_foul = [r"technical foul", FOUL, 1]
    misses_bank_shot = [r"misses (\d+)-foot jump bank shot|misses jump bank shot", POINT, 0]
    flagrant_foul_1 = [r"flagrant foul type 1", FOUL, 1]
    makes_ft_flagrant_1_of_2 = [r"makes free throw flagrant 1 of 2", POINT, 1]
    makes_ft_flagrant_2_of_2 = [r"makes free throw flagrant 2 of 2", POINT, 1]
    misses_ft_flagrant_1_of_2 = [r"misses free throw flagrant 1 of 2", POINT, 0]
    misses_ft_flagrant_2_of_2 = [r"misses free throw flagrant 2 of 2", POINT, 0]
    makes_ft_flagrant_1_of_3 = [r"makes free throw flagrant 1 of 3", POINT, 1]
    makes_ft_flagrant_2_of_3 = [r"makes free throw flagrant 2 of 3", POINT, 1]
    makes_ft_flagrant_3_of_3 = [r"makes free throw flagrant 3 of 3", POINT, 1]
    misses_ft_flagrant_1_of_3 = [r"misses free throw flagrant 1 of 3", POINT, 0]
    misses_ft_flagrant_2_of_3 = [r"misses free throw flagrant 2 of 3", POINT, 0]
    misses_ft_flagrant_3_of_3 = [r"misses free throw flagrant 3 of 3", POINT, 0]
    unspecified_foul = [r".*foul:.*", BOTH_TEAM_FOUL, 1]
    ejected = [r"ejected", POINT, 0]
    makes_running_jumper = [r"makes (\d+)-foot running jumper|makes running jumper", POINT, 2]
    misses_running_jumper = [r"misses (\d+)-foot running jumper|misses running jumper", POINT, 0]
    defensive_team_rebound = [r"defensive team rebound", REBOUND, 0]
    team_rebound = [r"team rebound", REBOUND, 0]
    lost_ball = [r"lost ball", TURNOVER, 1]






    play_types = [makes_three_point_jumper, misses_three_point_jumper,makes_two_point, misses_two_point, makes_pullup,
                  misses_pullup, blocks, defensive_rebound, turnover, shooting_foul, makes_free_throw_one_of_one, misses_free_throw_one_of_one,
                  makes_free_throw_one_of_two, makes_free_throw_two_of_two, makes_free_throw_one_of_three, makes_free_throw_two_of_three,
                  makes_free_throw_three_of_three,misses_free_throw_one_of_two, misses_free_throw_two_of_two, misses_free_throw_one_of_three,
                  misses_free_throw_two_of_three, misses_free_throw_three_of_three,offensive_rebound, makes_driving_layup, misses_driving_layup,
                  makes_layup, misses_layup, bad_pass, makes_driving_floating_jumpshot, misses_driving_floating_jumpshot,
                  makes_three_point_pullup, misses_three_point_pullup, personal_foul, makes_driving_dunk, misses_driving_dunk,
                  makes_alley_oop_dunk_shot, misses_alley_oop_dunk_shot, makes_running_pullup_jumpshot, misses_running_pullup_jumpshot,
                  makes_stepback_jumpshot, misses_stepback_jumpshot,makes_tip_shot,misses_tip_shot,makes_alley_oop_layup,
                  misses_alley_oop_layup,offensive_foul,loose_ball_foul,makes_dunk,misses_dunk, traveling, makes_bank_shot,
                  makes_hook_shot,misses_hook_shot, kicked_ball_violation, offensive_charge, violation, makes_finger_roll_layup,
                  misses_finger_roll_layup, personal_take_foul, transition_take_foul, defensive_three_seconds, makes_technical_free_throw,
                  misses_technical_free_throw, hanging_techfoul, technical_foul, misses_bank_shot, flagrant_foul_1, makes_ft_flagrant_1_of_2,
                  makes_ft_flagrant_2_of_2, misses_ft_flagrant_1_of_2, misses_ft_flagrant_2_of_2, makes_ft_flagrant_1_of_3,
                  makes_ft_flagrant_2_of_3,makes_ft_flagrant_3_of_3,misses_ft_flagrant_1_of_3,misses_ft_flagrant_2_of_3,misses_ft_flagrant_3_of_3,
                  unspecified_foul, ejected, makes_running_jumper, misses_running_jumper, defensive_team_rebound, team_rebound, lost_ball]

    for i, play_type in enumerate(play_types):
        match = re.search(play_type[0], play_info)
        if match:
            if play_type[1] == POINT or play_type[1] == REBOUND or play_type[1] == BLOCK or play_type[1] == STEAL or play_type[1] == FOUL\
                    or play_type[1] == TURNOVER:
                if player_name is not None:
                    box_score[player_name][play_type[1]] += play_type[2]
            if play_type[1] == FOUL:
                if team is not None:
                    team_fouls[team] += 1
            if play_type[1] == TURNOVER:
                if team is not None:
                    team_turnovers[team] += 1
            if play_type[1] == FOUL_AND_TURNOVER:
                if team is not None:
                    team_fouls[team] += 1
                    team_turnovers[team] += 1
            if play_type[1] == BOTH_TEAM_FOUL:
                team_fouls[HOME] += 1
                team_fouls[AWAY] += 1
            if play_type[1] == TEAM_FOUL:
                team_fouls[team] += 1
            if match.lastindex is None:
                return i*100
            else:
                for index in range(1, match.lastindex + 1):
                    if match.group(index) is not None:
                        dist = DIST_DEFAULT_FT
                        if int(match.group(index)) <= 4:
                            dist = DIST_WITHIN_4_FT
                        elif int(match.group(index)) >= 5 and int(match.group(index)) <= 14:
                            dist = DIST_5_TO_14
                        elif int(match.group(index)) >= 15 and int(match.group(index)) <= 26:
                            dist = DIST_15_26
                        elif int(match.group(index)) >= 27 and int(match.group(index)) <= 30:
                            dist = DIST_27_30
                        else:
                            dist = DIST_30_PLUS
                        return i * 100 + dist
    return -1


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





def get_roster(box_score_url):
    driver = webdriver.Chrome(options=chrome_options)

    # Open the page URL
    open_website_with_retry(box_score_url, driver)

    # Wait for the content to load
    driver.implicitly_wait(7)  # Adjust the sleep time as needed
    # Get the page source after the dynamic content is loaded
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table_selector = ('.GameBoxscore_gbTableSection__zTOUg')
    tables = soup.select(selector=table_selector)
    away_team_elements = tables[0].select(selector='.GameBoxscoreTablePlayer_gbpNameFull__cf_sn')
    home_team_elements = tables[1].select(selector='.GameBoxscoreTablePlayer_gbpNameFull__cf_sn')
    if not away_team_elements:
        print("ERROR ROSTER AWAY")
        exit(-1)
    if not home_team_elements:
        print("ERROR ROSTER HOME")
        exit(-1)
    player_names_home = [player.get_text(strip=True) for player in home_team_elements]
    player_names_away = [player.get_text(strip=True) for player in away_team_elements]
    driver.quit()
    return player_names_home, player_names_away




def get_play_component_data(page_url, stage, game_num, box_score_url):
    # Initialize WebDriver (make sure to specify the path to your WebDriver)
    global box_score
    global team_fouls
    global team_turnovers
    home_roster, away_roster = get_roster(box_score_url)
    driver = webdriver.Chrome(options=chrome_options)

    # Open the page URL
    open_website_with_retry(page_url, driver)

    # Wait for the content to load
    driver.implicitly_wait(7)  # Adjust the sleep time as needed
    # Get the page source after the dynamic content is loaded
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    teams = soup.find_all(True, class_=['fw-medium n7 ml2'])
    teams_text = [team.get_text(strip=True) for team in teams]
    records = soup.find_all(True, class_=['Gamestrip__Record db clr-gray-03 n9'])
    records_text = [record.get_text(strip=True) for record in records]
    team_fouls = [0,0]
    team_turnovers = [0,0]
    away_wins = int(records_text[0].split('-')[0])
    away_losses = int(records_text[0].split(",")[0].split('-')[1])
    away_win_percentage = round(away_wins/(away_wins + away_losses) * 100, 2)
    home_wins = int(records_text[1].split('-')[0])
    home_losses = int(records_text[1].split(",")[0].split('-')[1])
    home_win_percentage = round(home_wins/(home_wins + home_losses) * 100, 2)

    win_diff = abs(int(home_wins) - int(away_wins))

    home_team = teams_text[1]
    away_team = teams_text[0]

    play_components_text = []
    quarter_texts = ['1st', '2nd', '3rd', '4th', 'OT', '2 OT']


    driver.execute_script("window.scrollBy(0, 200);")
    init_array = [0, 0, 0, 0, 0, 0, 0] #rebounds, assists, steals, blocks, points, fouls, turnovers
    box_score = dict.fromkeys(home_roster+away_roster)
    for player in home_roster + away_roster:
        box_score[player] = deepcopy(init_array)


    for quarter_text in quarter_texts:
        team_fouls = [0,0]
        # Click on the quarter button
        existence_check = driver.find_elements(By.XPATH,
                                             f'//button[contains(@class, "Button--unstyled tabs__link") and text()'
                                             f'="{quarter_text}"]')
        if not existence_check:
            break
        quarter_button = existence_check[0]
        quarter_button.click()
        driver.implicitly_wait(6)  # Adjust the sleep time as needed
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        play_times_q = soup.find_all(True, class_=['playByPlay__time Table__TD'])
        play_info_q = soup.find_all(True, class_=['playByPlay__text tl Table__TD',
                                                  'playByPlay__text tl clr-btn Table__TD'])
        away_scores = soup.find_all(True, class_=['playByPlay__score playByPlay__score--away tr Table__TD',
                                                  'playByPlay__score playByPlay__score--away tr fw-normal Table__TD'])
        home_scores = soup.find_all(True, class_=['playByPlay__score playByPlay__score--home tr Table__TD',
                                                  'playByPlay__score playByPlay__score--home tr fw-normal Table__TD'])
        game_date_time_soup = soup.find_all(True, class_=['n8 GameInfo__Meta'])

        if not play_times_q:
            print("Parsing Error 1!" + quarter_text)
            return -1
        if not play_info_q:
            print("Parsing Error 2!" + quarter_text)
            return -1
        if not away_scores:
            print("Parsing Error 3!" + quarter_text)
        if not home_scores:
            print("Parsing Error 4!" + quarter_text)

        game_date_time = [element.get_text(strip=True) for element in game_date_time_soup]
        date_pattern = r"(\w+ \d{1,2}, \d{4})"
        match = re.search(date_pattern, game_date_time[0])
        if match:
            game_date = match.group(1)
        else:
            game_date = "Date not found"
        play_times_text = [play_time.get_text(strip=True) for play_time in play_times_q]
        play_info_text = [info.get_text(strip=True) for info in play_info_q]
        play_home_score_text = [score.get_text(strip=True) for score in home_scores]
        play_away_score_text = [score.get_text(strip=True) for score in away_scores]
        for play_time, play_info, home_score, away_score in zip(play_times_text, play_info_text, play_home_score_text, play_away_score_text):
            #print(play_info)
            if "vs" in play_info:
                continue
            splitted = play_info.split()
            player_name = "cannot identify"
            if (home_team in play_info.upper() or away_team in play_info.upper() or "enters" in play_info or "End of" in\
                    play_info or "Start of" in play_info or "delay" in play_info or "timeout" in play_info or "Game"
                    in play_info):
                player_name = "None"
                continue
            elif "Jr." == splitted[2] or "II" == splitted[2] or "III" == splitted[2] or "IV" == splitted[2] or "Sr." == splitted[2]:
                player_name = splitted[0] + " " + splitted[1] + " " + splitted[2]
                play_info = ' '.join(play_info.split()[3:])
            else:
                if splitted[0] == "O.G.":
                    player_name = "OG " + splitted[1]
                else:
                    player_name = splitted[0] + " " + splitted[1]
                play_info = ' '.join(play_info.split()[2:])

            assister = removeAssister(play_info)[0]
            assister = name_corrector(assister,home_roster, away_roster)
            if assister in home_roster or assister in away_roster:
                box_score[assister][ASSIST] += 1
                assister_assists = box_score[assister][ASSIST]
            else:
                assister_assists = "None"
            stealer = removeStealer(play_info)[0]
            stealer = name_corrector(stealer, home_roster, away_roster)
            if stealer in home_roster or stealer in away_roster:
                box_score[stealer][STEAL] += 1
                stealer_steals = box_score[stealer][STEAL]
            else:
                stealer_steals = "None"
            pattern = re.compile(r'\((.*?)\)')
            pattern_match = pattern.findall(play_info)
            if len(pattern_match) > 0:
                if 'assists' not in pattern_match[0] and 'steals' not in pattern_match[0]:
                    debug = 1
            play_info = re.sub(r'\(.*?\)', '', play_info).strip()
            player_name = name_corrector(player_name, home_roster, away_roster)
            if player_name in home_roster:
                play_info_id = playID(play_info, player_name, HOME)
            elif player_name in away_roster:
                play_info_id = playID(play_info, player_name, AWAY)
            else:
                play_info_id = playID(play_info, None, None)
                if player_name != "shot clock" and player_name != "5 second":
                    debug = 1

            if play_info_id == -1:
                play_info_id = play_info
            total_games = min(home_wins + home_losses, away_wins + away_losses)
            if player_name in home_roster:
                win_percentage = home_win_percentage
                fouls = team_fouls[HOME]
                turnovers = team_turnovers[HOME]
            elif player_name in away_roster:
                win_percentage = away_win_percentage
                fouls = team_fouls[AWAY]
                turnovers = team_turnovers[AWAY]
            else:
                win_percentage = "None"
                fouls = "None"
                turnovers = "None"
            if player_name in home_roster or player_name in away_roster:
                player_rebs = box_score[player_name][REBOUND]
                player_assists = box_score[player_name][ASSIST]
                player_steals = box_score[player_name][STEAL]
                player_blocks = box_score[player_name][BLOCK]
                player_points = box_score[player_name][POINT]
                player_fouls = box_score[player_name][FOUL]
                player_turnovers = box_score[player_name][TURNOVER]
            else:
                player_rebs = "None"
                player_assists = "None"
                player_steals = "None"
                player_blocks = "None"
                player_points = "None"
                player_fouls = "None"
                player_turnovers = "None"
            play_components_text.append([play_time, play_info_id, quarter_text, home_team, away_team, player_name,
                                         assister, stage, game_num, win_diff, total_games, win_percentage,
                                         home_score, away_score, abs(int(home_score) - int(away_score)), fouls, turnovers, player_rebs,
                                         player_assists, player_steals, player_blocks, player_points, player_fouls, player_turnovers, assister_assists, stealer, stealer_steals, game_date])
    if not play_components_text:
        print("No play components found. Verify the class name or structure of the HTML.")
    else:
        print(f"Found {len(play_components_text)} plays.")
    driver.quit()
    return play_components_text

def main():
    page_url = input("Enter AUTO to generate default. Or enter ESPN game urls manually and enter END to finish the "
                     "program\n")
    play_data = [["time", "play", "quarter", "home_team", "away_team", "name", "assister", "stage", "game_num",
                  "win_difference", "games_played", "win_percentage", "home_score", "away_score", "score_difference", "team_fouls_qtr",
                  "team_turnovers","player_rebounds","player_assists","player_steals","player_blocks","player_points", "player_fouls", "player_turnovers","assister_assists","stolen_by","stealer_steals","date"]]
    if page_url == "AUTO":
        with open('game_urls.txt', 'r') as file:
            for line in file:
                # Strip the newline character and any leading/trailing whitespace
                cleaned_line = line.strip()
                split_line = cleaned_line.split()
                print(line)
                stage = split_line[0]
                game_num = split_line[1]
                page_url = split_line[2]
                box_score_url = split_line[3]
                play_data += get_play_component_data(page_url, stage, game_num, box_score_url)

    else:
        while page_url != "END":
            box_score_url = input("Enter nba.com box score URL\n")
            stage = input("enter stage\n")
            game_num = input("enter game num\n")
            play_data += get_play_component_data(page_url, stage, game_num, box_score_url)
            if play_data == -1:
                print("FATAL ERROR")
                return
            #save_to_file(play_data, 'play_components.txt')
            page_url = input("Enter next espn URL or END to finish\n")
    with open('output_v3.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(play_data)
    print(f"Play component text saved to output_v3.csv")
    print("OK :)")

if __name__ == '__main__':
    main()

