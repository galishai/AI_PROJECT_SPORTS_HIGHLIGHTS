import os
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
from unidecode import unidecode
from termcolor import colored
import json

os.system('color')

SAVE_ROSTERS = 0
DEBUG = 0
ROSTER_DICT = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/temp_rosters.txt'
ESPN_LINKS = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/rs_espn_links.txt'
ESPN_LINKS_DEBUG = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/debug/espn_links_debug.txt'
NBA_COM_LINKS = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/rs_nbadotcom_links.txt'
NBA_COM_LINKS_DEBUG = '/Users/galishai/PycharmProjects/AI_PROJECT_SPORTS_HIGHLIGHTS/full season data/debug/nba_com_links_debug.txt'
IGNORE_PLAY_LIST = []

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('capabilities={"acceptInsecureCerts": True}')
#TODO CHECK IF NEEDED
chrome_options.add_argument("--disable-images")
chrome_options.add_argument("--disable-javascript")
driver = webdriver.Chrome(options=chrome_options)

class Plays(enum.Enum):
    MAKES_THREE_POINT_JUMPER = 0
    MISSES_THREE_POINT_JUMPER = 1
    MAKES_TWO_POINT = 2
    MISSES_TWO_POINT = 3
    MAKES_PULLUP = 4
    MISSES_PULLUP = 5
    BLOCK = 6
    DEFENSIVE_REBOUND = 7
    TURNOVER = 8
    SHOOTING_FOUL = 9
    MAKES_FREE_THROW_ONE_OF_ONE = 10
    MISSES_FREE_THROW_ONE_OF_ONE = 11
    MAKES_FREE_THROW_ONE_OF_TWO = 12
    MAKES_FREE_THROW_TWO_OF_TWO = 13
    MAKES_FREE_THROW_ONE_OF_THREE = 14
    MAKES_FREE_THROW_TWO_OF_THREE = 15
    MAKES_FREE_THROW_THREE_OF_THREE = 16
    MISSES_FREE_THROW_ONE_OF_TWO = 17
    MISSES_FREE_THROW_TWO_OF_TWO = 18
    MISSES_FREE_THROW_ONE_OF_THREE = 19
    MISSES_FREE_THROW_TWO_OF_THREE = 20
    MISSES_FREE_THROW_THREE_OF_THREE = 21
    OFFENSIVE_REBOUND = 22
    MAKES_DRIVING_LAYUP = 23
    MISSES_DRIVING_LAYUP = 24
    MAKES_LAYUP = 25
    MISSES_LAYUP = 26
    BAD_PASS = 27
    MAKES_DRIVING_FLOATING_JUMPSHOT = 28
    MISSES_DRIVING_FLOATING_JUMPSHOT = 29
    MAKES_THREE_POINT_PULLUP = 30
    MISSES_THREE_POINT_PULLUP = 31
    PERSONAL_FOUL = 32
    MAKES_DRIVING_DUNK = 33
    MISSES_DRIVING_DUNK = 34
    MAKES_ALLEY_OOP_DUNK_SHOT = 35
    MISSES_ALLEY_OOP_DUNK_SHOT = 36
    MAKES_RUNNING_PULLUP_JUMPSHOT = 37
    MISSES_RUNNING_PULLUP_JUMPSHOT = 38
    MAKES_STEPBACK_JUMPSHOT = 39
    MISSES_STEPBACK_JUMPSHOT = 40
    MAKES_TIP_SHOT = 41
    MISSES_TIP_SHOT = 42
    MAKES_ALLEY_OOPS_LAYUP = 43
    MISSES_ALLEY_OOPS_LAYUP = 44
    OFFENSIVE_FOUL = 45
    LOOSE_BALL_FOUL = 46
    MAKES_DUNK = 47
    MISSES_DUNK = 48
    TRAVELING = 49
    MAKES_BANK_SHOT = 50
    MAKES_HOOK_SHOT = 51
    MISSES_HOOK_SHOT = 52
    KICKED_BALL_VIOLATION = 53
    OFFENSIVE_CHARGE = 54
    VIOLATION = 55
    MAKES_FINGER_ROLL_LAYUP = 56
    MISSES_FINGER_ROLL_LAYUP = 57
    PERSONAL_TAKE_FOUL = 58
    TRANSITION_TAKE_FOUL = 59
    DEFENSIVE_THREE_SECONDS = 60
    MAKES_TECHNICAL_FREE_THROW = 61
    MISSES_TECHNICAL_FREE_THROW = 62
    HANGING_TECHFOUL = 63
    TECHNICAL_FOUL = 64
    MISSES_BANK_SHOT = 65
    FLAGRANT_FOUL_1 = 66
    MAKES_FREE_THROW_FLAGRANT_1_OF_2 = 67
    MAKES_FREE_THROW_FLAGRANT_2_OF_2 = 68
    MISSES_FREE_THROW_FLAGRANT_1_OF_2 = 69
    MISSES_FREE_THROW_FLAGRANT_2_OF_2 = 70
    MAKES_FREE_THROW_FLAGRANT_1_OF_3 = 71
    MAKES_FREE_THROW_FLAGRANT_2_OF_3 = 72
    MAKES_FREE_THROW_FLAGRANT_3_OF_3 = 73
    MISSES_FREE_THROW_FLAGRANT_1_OF_3 = 74
    MISSES_FREE_THROW_FLAGRANT_2_OF_3 = 75
    MISSES_FREE_THROW_FLAGRANT_3_OF_3 = 76
    BOTH_TEAM_FOUL = 77
    EJECTED = 78
    MAKES_RUNNING_JUMPER = 79
    MISSES_RUNNING_JUMPER = 80
    DEFENSIVE_TEAM_REBOUND = 81
    TEAM_REBOUND = 82
    LOST_BALL = 83
    AWAY_FROM_PLAY_FOUL = 84
    UNSPECIFIED_FOUL = 85
    MAKES_FREE_THROW_FLAGRANT_1_OF_1 = 86
    UNSPECIFIED_SHOT_CLOCK_TO = 87

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
nba_team_names = ['CELTICS', 'MAVERICKS', 'NUGGETS', 'MAGIC', 'KNICKS', 'ROCKETS', 'PISTONS', 'JAZZ', 'WIZARDS',
                  'WARRIORS', 'SPURS', 'HAWKS', 'RAPTORS', 'GRIZZLIES', 'SUNS', 'PACERS', 'HORNETS', 'NETS', 'CLIPPERS',
                  'THUNDER', 'KINGS', 'TRAIL BLAZERS', 'LAKERS', 'HEAT', '76ERS', 'BULLS', 'BUCKS', 'CAVALIERS', 'PELICANS',
                  'TIMBERWOLVES']

nba_team_logos = ['bos','dal','den','orl','ny','hou','det','utah','wsh','gs','sa','atl','tor','mem','phx','ind','cha','bkn',
                  'lac','okc','sac','por','lal','mia','phi','chi','mil','cle','no','min']

nba_teams_to_logos = {}
for name, logo in zip(nba_team_names,nba_team_logos):
    nba_teams_to_logos[name] = logo + '.png'

team_rosters_full = {}
player_vector_size = 0

#returns player that assisted, or "None" if unassisted
def removeAssister(play_info):

    assist_pattern = re.compile(rf'\((.*?)\s+assists?\)')
    assist_match = assist_pattern.findall(play_info)
    if not assist_match:
        return ["Blank"]
    else:
        return assist_match

def removeStealer(play_info):
    steal_pattern = re.compile(rf'\((.*?)\s+steals?\)')
    steal_match = steal_pattern.findall(play_info)
    if not steal_match:
        return ["Blank"]
    else:
        return steal_match

def jr_ify(player_name):
    return player_name + " Jr."

def sr_ify(player_name):
    return player_name + " Sr."

def name_corrector(player_name, home_roster, away_roster):

    if jr_ify(player_name) in home_roster or jr_ify(player_name) in away_roster:
        return jr_ify(player_name)
    if sr_ify(player_name) in home_roster or sr_ify(player_name) in away_roster:
        return sr_ify(player_name)
    if player_name == 'Jeenathan Williams':
        return 'Nate Williams'
    if player_name == "Amare Stoudemire":
        return "Amar'e Stoudemire"
    if player_name == "O.G. Anunoby":
        return "OG Anunoby"
    if player_name == "Bill Walker":
        return "Henry Walker"
    if player_name == "Jakob Poltl":
        return "Jakob Poeltl"
    if player_name == "Trey Jemison III":
        return "Trey Jemison"
    if player_name == "Xavier Tillman Sr.":
        return "Xavier Tillman"
    if player_name == "Kenyon Martin Jr.":
        return "KJ Martin"
    if player_name == "Matthew Hurt":
        return "Matt Hurt"
    return unidecode(player_name)

def roster_fix(roster):
    for player in roster:
        roster[roster.index(player)] = name_corrector(player, roster, roster)
    return roster

#returns numerical value of play, distance. FORMAT: Regex, STAT, VALUE
def playID(play_info, player_name, team, home_score, away_score, prev_home_score, prev_away_score, curr_team):
    global box_score
    global team_fouls
    global team_turnovers
    makes_three_point_jumper = [r"makes (\d+)-foot three point jumper|makes (\d+)-foot three point shot|makes (\d+)-foot three pointer|makes three point jumper|makes three point shot|makes three pointer", POINT]
    misses_three_point_jumper = [r"misses (\d+)-foot three point jumper|misses (\d+)-foot three point shot|misses (\d+)-foot three pointer|misses three point jumper|misses three point shot|misses three pointer", POINT]
    makes_two_point = [r"makes (\d+)-foot two point shot|makes (\d+)-foot jumper|makes two point shot|makes jumper|makes (\d+)-foot shot", POINT]
    misses_two_point = [r"misses (\d+)-foot two point shot|misses (\d+)-foot jumper|misses two point shot|misses jumper|misses shot", POINT]
    makes_pullup = [r"makes (\d+)-foot pullup jump shot|makes pullup jump shot", POINT]
    misses_pullup = [r"misses (\d+)-foot pullup jump shot|misses pullup jump shot|misses (\d+)-foot jumpshot", POINT]
    blocks= [r"^blocks", BLOCK]
    defensive_rebound = [r"defensive rebound", REBOUND]
    turnover = [r"\bturnover\b", TURNOVER]
    shooting_foul =  [r"shooting foul", FOUL]
    makes_free_throw_one_of_one = [r"makes free throw 1 of 1", POINT]
    misses_free_throw_one_of_one = [r"misses free throw 1 of 1", POINT]
    makes_free_throw_one_of_two = [r"makes free throw 1 of 2|makes free throw clear path 1 of 2", POINT]
    makes_free_throw_two_of_two = [r"makes free throw 2 of 2|makes free throw clear path 2 of 2", POINT]
    makes_free_throw_one_of_three = [r"makes free throw 1 of 3", POINT]
    makes_free_throw_two_of_three = [r"makes free throw 2 of 3", POINT]
    makes_free_throw_three_of_three = [r"makes free throw 3 of 3", POINT]
    misses_free_throw_one_of_two = [r"misses free throw 1 of 2|misses free throw clear path 1 of 2", POINT]
    misses_free_throw_two_of_two = [r"misses free throw 2 of 2|misses free throw clear path 2 of 2", POINT]
    misses_free_throw_one_of_three = [r"misses free throw 1 of 3", POINT]
    misses_free_throw_two_of_three = [r"misses free throw 2 of 3", POINT]
    misses_free_throw_three_of_three = [r"misses free throw 3 of 3", POINT]
    offensive_rebound = [r"offensive rebound", REBOUND]
    makes_driving_layup = [r"makes driving layup", POINT]
    misses_driving_layup = [r"misses driving layup", POINT]
    makes_layup = [r"makes layup|makes (\d+)-foot layup", POINT]
    misses_layup = [r"misses layup|misses (\d+)-foot layup", POINT]
    bad_pass = [r"bad pass", TURNOVER]
    makes_driving_floating_jumpshot = [r"makes (\d+)-foot driving floating jump shot|makes driving floating jump shot", POINT]
    misses_driving_floating_jumpshot = [r"misses (\d+)-foot driving floating jump shot|misses driving floating jump shot", POINT]
    makes_three_point_pullup = [r"makes (\d+)-foot three point pullup jump shot|makes three point pullup jump shot", POINT]
    misses_three_point_pullup = [r"misses (\d+)-foot three point pullup jump shot|misses three point pullup jump shot", POINT]
    personal_foul = [r"personal foul", FOUL]
    makes_driving_dunk = [r"makes (\d+)-foot driving dunk|makes driving dunk", POINT]
    misses_driving_dunk = [r"misses (\d+)-foot driving dunk|misses driving dunk", POINT]
    makes_alley_oop_dunk_shot = [r"makes (\d+)-foot alley oop dunk shot|makes alley oop dunk shot", POINT]
    misses_alley_oop_dunk_shot = [r"makes (\d+)-foot alley oop dunk shot|misses alley oop dunk shot", POINT]
    makes_running_pullup_jumpshot = [r"makes (\d+)-foot running pullup jump shot|makes running pullup jump shot", POINT]
    misses_running_pullup_jumpshot = [r"misses (\d+)-foot running pullup jump shot|misses running pullup jump shot", POINT]
    makes_stepback_jumpshot = [r"makes (\d+)-foot step back jumpshot|makes step back jumpshot", POINT]
    misses_stepback_jumpshot = [r"misses (\d+)-foot step back jumpshot|misses step back jumpshot", POINT]
    makes_tip_shot = [r"makes (\d+)-foot tip shot|makes tip shot", POINT]
    misses_tip_shot = [r"misses (\d+)-foot tip shot|misses tip shot", POINT]
    makes_alley_oop_layup = [r"makes alley oop layup", POINT]
    misses_alley_oop_layup = [r"misses alley oop layup", POINT]
    offensive_foul = [r"offensive foul", FOUL]
    loose_ball_foul = [r"loose ball foul", FOUL]
    makes_dunk = [r"makes (\d+)-foot dunk|makes dunk|makes slam dunk", POINT]
    misses_dunk = [r"misses (\d+)-foot dunk|misses dunk|misses slam dunk", POINT]
    traveling = [r"traveling|raveling", TURNOVER]
    makes_bank_shot = [r"makes (\d+)-foot jump bank shot|makes jump bank shot", POINT]
    makes_hook_shot = [r"makes (\d+)-foot hook shot|makes hook shot", POINT]
    misses_hook_shot = [r"misses (\d+)-foot hook shot|misses hook shot", POINT]
    kicked_ball_violation = [r"kicked ball violation", TURNOVER]
    offensive_charge = [r"offensive charge", FOUL_AND_TURNOVER]
    violation = [r"violation", TURNOVER]
    makes_finger_roll_layup = [r"makes finger roll layup", POINT]
    misses_finger_roll_layup = [r"misses finger roll layup", POINT]
    personal_take_foul = [r"personal take foul", FOUL]
    transition_take_foul = [r"transition take foul", FOUL]
    defensive_three_seconds = [r"defensive 3-seconds", TEAM_FOUL]
    makes_technical_free_throw = [r"makes technical free throw", POINT]
    misses_technical_free_throw = [r"misses technical free throw", POINT]
    hanging_techfoul = [r"hanging techfoul", FOUL]
    technical_foul = [r"technical foul|Players Technical|defensive 3-seconds (technical foul)", FOUL]
    misses_bank_shot = [r"misses (\d+)-foot jump bank shot|misses jump bank shot", POINT]
    flagrant_foul_1 = [r"flagrant foul type 1", FOUL]
    makes_ft_flagrant_1_of_2 = [r"makes free throw flagrant 1 of 2", POINT]
    makes_ft_flagrant_2_of_2 = [r"makes free throw flagrant 2 of 2", POINT]
    misses_ft_flagrant_1_of_2 = [r"misses free throw flagrant 1 of 2", POINT]
    misses_ft_flagrant_2_of_2 = [r"misses free throw flagrant 2 of 2", POINT]
    makes_ft_flagrant_1_of_3 = [r"makes free throw flagrant 1 of 3", POINT]
    makes_ft_flagrant_2_of_3 = [r"makes free throw flagrant 2 of 3", POINT]
    makes_ft_flagrant_3_of_3 = [r"makes free throw flagrant 3 of 3", POINT]
    misses_ft_flagrant_1_of_3 = [r"misses free throw flagrant 1 of 3", POINT]
    misses_ft_flagrant_2_of_3 = [r"misses free throw flagrant 2 of 3", POINT]
    misses_ft_flagrant_3_of_3 = [r"misses free throw flagrant 3 of 3", POINT]
    both_team_foul = [r".*foul:.*", BOTH_TEAM_FOUL]
    ejected = [r"ejected", POINT]
    makes_running_jumper = [r"makes (\d+)-foot running jumper|makes running jumper", POINT]
    misses_running_jumper = [r"misses (\d+)-foot running jumper|misses running jumper", POINT]
    defensive_team_rebound = [r"defensive team rebound", REBOUND]
    team_rebound = [r"team rebound", REBOUND]
    lost_ball = [r"lost ball", TURNOVER]
    away_from_play_foul = [r"away from play foul", FOUL]
    unspecified_foul = [r".*foul.*|Too Many Players Technical", FOUL]
    makes_ft_flagrant_1_of_1 = [r"makes free throw flagrant 1 of 1", POINT]
    misses_ft_flagrant_1_of_1 = [r"misses free throw flagrant 1 of 1", POINT]
    unspecified_shot_clock_to = [r"shot clock turnover", TURNOVER]






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
                  both_team_foul, ejected, makes_running_jumper, misses_running_jumper, defensive_team_rebound, team_rebound, lost_ball, away_from_play_foul, unspecified_foul, makes_ft_flagrant_1_of_1, misses_ft_flagrant_1_of_1, unspecified_shot_clock_to]

    for i, play_type in enumerate(play_types):
        match = re.search(play_type[0], play_info)
        if match:
            if play_type[1] == POINT:
                if play_info == 'ejected':
                    return i, DIST_DEFAULT_FT
                if team == HOME:
                    box_score[player_name][POINT] += int(home_score) - prev_home_score
                elif team == AWAY:
                    box_score[player_name][POINT] += int(away_score) - prev_away_score

            if play_type[1] == REBOUND or play_type[1] == BLOCK or play_type[1] == STEAL or play_type[1] == FOUL\
                    or play_type[1] == TURNOVER:
                if player_name not in team_rosters_full[curr_team]:
                    if play_type[1] != FOUL and play_type[1] != TURNOVER:
                        return -1, 0
                else:
                    box_score[player_name][play_type[1]] += 1
            if play_type[1] == FOUL:
                    team_fouls[team] += 1
            if play_type[1] == TURNOVER:
                    team_turnovers[team] += 1
            if play_type[1] == FOUL_AND_TURNOVER:
                    team_fouls[team] += 1
                    team_turnovers[team] += 1
            if play_type[1] == BOTH_TEAM_FOUL:
                team_fouls[HOME] += 1
                team_fouls[AWAY] += 1
            if play_type[1] == TEAM_FOUL:
                    team_fouls[team] += 1
            if match.lastindex is None:
                return i, DIST_DEFAULT_FT
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
                        elif int(match.group(index)) >= 31:
                            dist = DIST_30_PLUS
                        return i, dist
    print(play_info)
    return -1, 0

def has_name_play(play_info):
    if play_info == "defensive 3-seconds (technical foul)":
        return False
    if play_info == 'shot clock turnover':
        return False
    if play_info == 'technical foul':
        return False
    return True

def open_website_with_retry(url, driver, retries=3, delay=3):
    attempt = 0

    while attempt < retries:
        try:
            driver.get(url)

            # Check if the page loaded successfully by checking the title or page content
            if "502 Bad Gateway" not in driver.page_source:
                #print("Page loaded successfully!")
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
        return -1
    else:
        print("Website loaded successfully.")

    return driver

def binary_vector_to_decimal(binary_vector):
    decimal = 0
    for i, bit in enumerate(reversed(binary_vector)):
        decimal += bit * (2**i)
    return decimal






def get_roster(box_score_url):
    global team_rosters_full
    #driver = webdriver.Chrome(options=chrome_options)
    ret = open_website_with_retry(box_score_url, driver)
    if ret == -1:
        return -1

    #driver.implicitly_wait(7)

    #time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table_selector = ('.GameBoxscore_gbTableSection__zTOUg')
    tables = soup.select(selector=table_selector)
    away_team_elements = tables[0].select(selector='.GameBoxscoreTablePlayer_gbpNameFull__cf_sn')
    home_team_elements = tables[1].select(selector='.GameBoxscoreTablePlayer_gbpNameFull__cf_sn')
    team_name_elements = soup.select(selector = '.GameBoxscoreTeamHeader_gbt__b9B6g')
    team_names = [unidecode(team.get_text(strip=True)) for team in team_name_elements]
    away_team_name = [word.upper() for word in team_names[0].split()]
    home_team_name = [word.upper() for word in team_names[1].split()]

    if not away_team_elements:
        print("ERROR ROSTER AWAY")
        exit(-1)
    if not home_team_elements:
        print("ERROR ROSTER HOME")
        exit(-1)
    player_names_home = [unidecode(player.get_text(strip=True)) for player in home_team_elements]
    player_names_away = [unidecode(player.get_text(strip=True)) for player in away_team_elements]
    player_names_home = roster_fix(player_names_home)
    player_names_away = roster_fix(player_names_away)
    num_teams_found = 0
    for team in nba_team_names:
        if team in away_team_name or (team == "TRAIL BLAZERS" and 'TRAIL' in away_team_name):
            if team in team_rosters_full:
                team_rosters_full[team] = list(set(team_rosters_full[team] + player_names_away))
            else:
                team_rosters_full[team] = player_names_away
            away_team_short = team
            num_teams_found += 1
        if team in home_team_name or (team == "TRAIL BLAZERS" and 'TRAIL' in home_team_name):
            if team in team_rosters_full:
                team_rosters_full[team] = list(set(team_rosters_full[team] + player_names_home))
            else:
                team_rosters_full[team] = player_names_home
            home_team_short = team
            num_teams_found += 1
    assert num_teams_found == 2, " ".join(home_team_name) + ", " + " ".join(away_team_name)
    if home_team_short == away_team_short:
        a=1
    #driver.quit()
    return player_names_home, player_names_away, home_team_short, away_team_short

def get_team_from_logo(curr_logo):
    for i, logo_t in enumerate(nba_team_logos):
        if logo_t + '.png' in curr_logo:
            return nba_team_names[i]



def get_play_component_data(page_url, starting_5s, roster_teams):

    global box_score
    global team_fouls
    global team_turnovers
    global nba_teams_to_logos
    #home_roster, away_roster = get_roster(box_score_url)
    #driver = webdriver.Chrome(options=chrome_options)

    ret = open_website_with_retry(page_url, driver)
    if ret == -1:
        return -1

    #driver.implicitly_wait(7)

    #time.sleep(1)
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
    if home_team != roster_teams[0] or away_team != roster_teams[1]:
        a = "team mismatch"
    game_team_dict = {home_team: HOME, away_team: AWAY}
    home_roster = team_rosters_full[home_team]
    away_roster = team_rosters_full[away_team]
    home_starting_5 = starting_5s[0]
    away_starting_5 = starting_5s[1]

    play_components_text = []
    quarter_texts = ['1st', '2nd', '3rd', '4th', 'OT', '2 OT']


    driver.execute_script("window.scrollBy(0, 200);")
    init_array = [0, 0, 0, 0, 0, 0, 0] #rebounds, assists, steals, blocks, points, fouls, turnovers
    box_score = dict.fromkeys(home_roster+away_roster)
    for player in home_roster + away_roster:
        box_score[player] = deepcopy(init_array)

    prev_home_score = 0
    prev_away_score = 0
    players_on_court_home = [0] * player_vector_size
    players_on_court_away = [0] * player_vector_size
    for player_home, player_away in zip(home_starting_5, away_starting_5):
        players_on_court_home[home_roster.index(player_home)] = 1
        players_on_court_away[away_roster.index(player_away)] = 1

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
        #driver.implicitly_wait(6)  # Adjust the sleep time as needed
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        play_times_q = soup.find_all(True, class_=['playByPlay__time Table__TD'])
        play_info_q = soup.find_all(True, class_=['playByPlay__text tl Table__TD',
                                                  'playByPlay__text tl clr-btn Table__TD'])
        away_scores = soup.find_all(True, class_=['playByPlay__score playByPlay__score--away tr Table__TD',
                                                  'playByPlay__score playByPlay__score--away tr fw-normal Table__TD'])
        home_scores = soup.find_all(True, class_=['playByPlay__score playByPlay__score--home tr Table__TD',
                                                  'playByPlay__score playByPlay__score--home tr fw-normal Table__TD'])
        game_date_time_soup = soup.find_all(True, class_=['n8 GameInfo__Meta'])
        logos = soup.find_all('td', class_='playByPlay__logo Table__TD')

        if not play_times_q:
            print("Parsing Error 1!" + quarter_text)
            return -1
        if not play_info_q:
            print("Parsing Error 2!" + quarter_text)
            return -1
        if not away_scores:
            print("Parsing Error 3!" + quarter_text)
            return -1
        if not home_scores:
            print("Parsing Error 4!" + quarter_text)
            return -1
        if not logos:
            print("Parsing Error 5!" + quarter_text)
            return -1
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
        for play_time, play_info, home_score, away_score, curr_logo in zip(play_times_text, play_info_text, play_home_score_text, play_away_score_text, logos):
            if play_info in IGNORE_PLAY_LIST:
                continue
            curr_logo = curr_logo.find('img')  # Check if <img> tag exists inside <td>
            if not curr_logo:
                #print("no logo found: " + play_info + '\n')
                continue
            else:
                curr_logo = curr_logo['src']
                if 'gif' in curr_logo:
                    continue
            #print(play_info)
            curr_team = get_team_from_logo(curr_logo)
            if curr_team != home_team and curr_team != away_team:
                assert 0 == 1, curr_team
            play_info_copy = play_info
            has_name = has_name_play(play_info)
            if "vs" in play_info or play_info == '':
                continue
            splitted = play_info.split()
            player_name = "cannot identify"
            if (home_team in play_info.upper() or away_team in play_info.upper() or "End of" in\
                    play_info or "Start of" in play_info or "delay" in play_info or "timeout" in play_info or "Game"
                    in play_info):
                continue
            if "enters" in play_info:
                player_in = splitted[0]
                player_in_arr = splitted[1:splitted.index("enters")]
                for w in player_in_arr:
                    player_in = player_in + ' ' + w
                try:
                    player_out = splitted[splitted.index("for") + 1]
                except:
                    continue
                player_out_arr = splitted[splitted.index("for") + 2:]
                for w in player_out_arr:
                    player_out = player_out + ' ' + w
                player_in = name_corrector(player_in, home_roster, away_roster)
                player_out = name_corrector(player_out, home_roster, away_roster)
                if not ((player_in in home_roster and player_out in home_roster) or (
                            player_in in away_roster and player_out in away_roster)):
                    print("player in: " + player_in)
                    print("player out: " + player_out)
                    print(home_roster)
                    print(away_roster)
                    assert(1 == 0)
                if nba_teams_to_logos[home_team] in curr_logo:
                    players_on_court_home[home_roster.index(player_in)] = 1
                    players_on_court_home[home_roster.index(player_out)] = 0
                else:
                    players_on_court_away[away_roster.index(player_in)] = 1
                    players_on_court_away[away_roster.index(player_out)] = 0
                continue


            if has_name and len(splitted) >= 2:
                if "Jr." == splitted[2] or "II" == splitted[2] or "III" == splitted[2] or "IV" == splitted[2] or "Sr." == splitted[2]:
                    player_name = splitted[0] + " " + splitted[1] + " " + splitted[2]
                    play_info = ' '.join(play_info.split()[3:])
                else:
                    player_name = splitted[0] + " " + splitted[1]
                    play_info = ' '.join(play_info.split()[2:])
            elif not has_name:
                player_name = "Blank"
            assister = removeAssister(play_info)[0]
            assister = name_corrector(assister,home_roster, away_roster)
            if assister in home_roster or assister in away_roster:
                box_score[assister][ASSIST] += 1
                assister_assists = box_score[assister][ASSIST]
            else:
                assister_assists = 0
            stealer = removeStealer(play_info)[0]
            stealer = name_corrector(stealer, home_roster, away_roster)
            if stealer in home_roster or stealer in away_roster:
                box_score[stealer][STEAL] += 1
                stealer_steals = box_score[stealer][STEAL]
            else:
                stealer_steals = 0
            pattern = re.compile(r'\((.*?)\)')
            pattern_match = pattern.findall(play_info)
            if len(pattern_match) > 0:
                if 'assists' not in pattern_match[0] and 'steals' not in pattern_match[0]:
                    debug = 1
            play_info = re.sub(r'\(.*?\)', '', play_info).strip()
            player_name = name_corrector(player_name, home_roster, away_roster)
            if 'blocks' in play_info:
                if curr_team == home_team:
                    curr_team = away_team
                else:
                    curr_team = home_team

            play_info_id, dist = playID(play_info, player_name, game_team_dict[curr_team], home_score, away_score, prev_home_score, prev_away_score, curr_team)


            if play_info_id == -1:
                print(play_info_copy)
                a=1
                continue
                #assert 1==0, "unrecognized play: " + play_info +'\n'
            total_games = min(home_wins + home_losses, away_wins + away_losses)
            if curr_team == home_team:
                win_percentage = home_win_percentage
                fouls = team_fouls[HOME]
                turnovers = team_turnovers[HOME]
                total_games = home_wins + home_losses #if stage == 'RS'
            elif curr_team == away_team:
                curr_team = away_team
                win_percentage = away_win_percentage
                fouls = team_fouls[AWAY]
                turnovers = team_turnovers[AWAY]
                total_games = away_wins + away_losses #if stage == 'RS':
            else:
                assert 1 == 0, curr_team
            if has_name and (player_name in home_roster or player_name in away_roster):
                player_rebs = box_score[player_name][REBOUND]
                player_assists = box_score[player_name][ASSIST]
                player_steals = box_score[player_name][STEAL]
                player_blocks = box_score[player_name][BLOCK]
                player_points = box_score[player_name][POINT]
                player_fouls = box_score[player_name][FOUL]
                player_turnovers = box_score[player_name][TURNOVER]
            else:
                player_rebs = 0
                player_assists = 0
                player_steals = 0
                player_blocks = 0
                player_points = 0
                player_fouls = 0
                player_turnovers = 0

            if curr_team == home_team:
                poc = players_on_court_home
            elif curr_team == away_team:
                poc = players_on_court_away
            else:
                assert 0 == 1, curr_team
                #poc = [0]*player_vector_size

            num_starters_playing = sum(poc[:5])

            prev_home_score = int(home_score)
            prev_away_score = int(away_score)
            play_components_text.append([play_time, play_info_id, dist, quarter_text, home_team, away_team,curr_team, player_name,
                                         assister, win_diff, total_games, win_percentage,
                                         home_score, away_score, fouls, turnovers, player_rebs,
                                         player_assists, player_steals, player_blocks, player_points, player_fouls,
                                         player_turnovers, assister_assists, stealer, stealer_steals, binary_vector_to_decimal(poc), game_date])
    if not play_components_text:
        print("No play components found. Verify the class name or structure of the HTML.")
        return -1
    else:
        print(f"Found {len(play_components_text)} plays.")
        assert len(play_components_text) > 250, 'too little plays found'

    #driver.quit()
    return play_components_text

def main():
    global player_vector_size, driver, team_rosters_full

    with open('../old_outputs/output_full_season_v1.csv', mode='w', newline='') as data_csv_file:
        writer = csv.writer(data_csv_file)
        writer.writerows([["time_left_qtr", "play", "distance", "quarter", "home_team", "away_team", "current_team", "name", "assister",
                      "win_difference", "games_played", "win_percentage", "home_score", "away_score", "team_fouls_qtr",
                      "team_turnovers","player_rebounds","player_assists","player_steals","player_blocks","player_points",
                      "player_fouls", "player_turnovers","assister_assists","stolen_by","stealer_steals", "players_binary_vector", "date"]])

    espn_page_url = input("Enter AUTO to generate default. Or enter ESPN game urls manually and enter END to finish the "
                     "program\n")
    espn_page_urls_arr = []
    nba_com_urls_arr = []
    stage_arr = []
    game_num_arr = []
    starting_5s_arr = []
    roster_team_order = []
    games_input_count = 0
    if espn_page_url == "AUTO":
        if DEBUG == 0:
            espn_links_path = ESPN_LINKS
            nba_com_links_path = NBA_COM_LINKS
        else:
            espn_links_path = ESPN_LINKS_DEBUG
            nba_com_links_path = NBA_COM_LINKS_DEBUG
        with open(espn_links_path, 'r') as espn_links:
            with open(nba_com_links_path, 'r') as nba_com_links:
                for espn_link, nba_com_link in zip(espn_links,nba_com_links):
                    # Strip the newline character and any leading/trailing whitespace
                    cleaned_espn, cleaned_nba_com = espn_link.strip(), nba_com_link.strip()
                    if cleaned_espn.startswith('#') or cleaned_nba_com.startswith('#'):
                        continue
                    #split_espn, split_nba_com = cleaned_espn.split(), cleaned_nba_com.split()
                    #stage_arr.append(split_line[0])
                    #game_num_arr.append(split_line[1])
                    espn_page_urls_arr.append(cleaned_espn)
                    nba_com_urls_arr.append(cleaned_nba_com)
                    games_input_count += 1
    else:
        while espn_page_url != "END":
            nba_com_url = input("Enter nba.com box score URL\n")
            #stage = input("enter stage\n")
            #game_num = input("enter game num\n")
            espn_page_urls_arr.append(espn_page_url)
            nba_com_urls_arr.append(nba_com_url)
            #stage_arr.append(stage)
            #game_num_arr.append(game_num)
            espn_page_url = input("Enter next espn URL or END to finish\n")
            games_input_count += 1
    assert len(espn_page_urls_arr) == len(nba_com_urls_arr)
    print(colored("valid game link pairs provided: " + str(games_input_count) + '\n','green'))
    #getting nba rosters
    rosters_processed_count = 0
    for bs_url in nba_com_urls_arr:
        try:
            starting_5_home, starting_5_away, home_team_name, away_team_name = get_roster(bs_url)
        except:
            driver.quit()
            driver = webdriver.Chrome(options=chrome_options)
            print(colored("FATAL ERROR, trying again... \n",'red'))
            try:
                starting_5_home, starting_5_away, home_team_name, away_team_name = get_roster(bs_url)
            except:
                print(colored("FATAL ERROR, link: " + bs_url + '\n','red'))
                driver.quit()
                return
            print(colored("recovered from fatal error\n", 'green'))
        starting_5s_arr.append([starting_5_home[:5], starting_5_away[:5]])
        roster_team_order.append([home_team_name,away_team_name])
        rosters_processed_count += 1
        if rosters_processed_count % 20 == 0:
            print(colored("roster links processed: " + str(rosters_processed_count) + '\n','green'))
            driver.quit()
            driver = webdriver.Chrome(options=chrome_options)
        #with open('temp_rosters.txt','w') as f:
        #    f.write(json.dumps(team_rosters_full))
    #with open(ROSTER_DICT) as f:
        #data = f.read()
    #team_rosters_full = json.loads(data)#
    player_vector_size = max(len(team_rosters_full[team]) for team in team_rosters_full)
    games_processed_count = 0
    with open('../old_outputs/output_full_season_v1.csv', mode='a', newline='') as data_csv_file:
        writer = csv.writer(data_csv_file)
        for espn_url, bs_url, starting_5s, roster_teams in zip(espn_page_urls_arr, nba_com_urls_arr, starting_5s_arr, roster_team_order):
            print("espn_page_url: " + espn_url + ", bs_url " + bs_url)
            component = get_play_component_data(espn_url, starting_5s, roster_teams)
            if component == -1:
                print(colored("FATAL ERROR ESPN, trying again... \n",'red'))
                driver.quit()
                driver = webdriver.Chrome(options=chrome_options)
                component = get_play_component_data(espn_url, starting_5s, roster_teams)
                if component == -1:
                    print(colored("FATAL ERROR ESPN. link :" + espn_url + '\n','red'))
                    driver.quit()
                    return
                print(colored("recovered from fatal error\n", 'green'))
            writer.writerows(component)
            games_processed_count += 1
            if games_processed_count % 20 == 0:
                print(colored("games processed: " + str(games_processed_count) + '\n','green'))
                driver.quit()
                driver = webdriver.Chrome(options=chrome_options)
        print(colored("games processed: " + str(games_processed_count) + '\n','green'))
        driver.quit()
    print(colored(f"Play component text saved to output_full_season_v1.csv",'green'))
    print(colored("OK :)",'green'))
    if not SAVE_ROSTERS:
        os.remove('temp_rosters.txt')

if __name__ == '__main__':
    main()

