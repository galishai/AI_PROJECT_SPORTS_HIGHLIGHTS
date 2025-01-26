from parsing.id_info.espn_plays import *

makes_three_point_jumper = [
        r"makes (\d+)-foot three point jumper"
        r"|makes (\d+)-foot three point shot"
        r"|makes (\d+)-foot three pointer"
        r"|makes three point jumper"
        r"|makes three point shot|makes three pointer",
        POINT]
misses_three_point_jumper = [
    r"misses (\d+)-foot three point jumper"
    r"|misses (\d+)-foot three point shot"
    r"|misses (\d+)-foot three pointer"
    r"|misses three point jumper"
    r"|misses three point shot"
    r"|misses three pointer",
    POINT]
makes_two_point = [
    r"makes (\d+)-foot two point shot"
    r"|makes (\d+)-foot jumper"
    r"|makes two point shot"
    r"|makes jumper"
    r"|makes (\d+)-foot shot",
    POINT]
misses_two_point = [
    r"misses (\d+)-foot two point shot"
    r"|misses (\d+)-foot jumper"
    r"|misses two point shot"
    r"|misses jumper"
    r"|misses shot",
    POINT]
makes_pullup = [r"makes (\d+)-foot pullup jump shot"
                r"|makes pullup jump shot",
                POINT]
misses_pullup = [
    r"misses (\d+)-foot pullup jump shot"
    r"|misses pullup jump shot"
    r"|misses (\d+)-foot jumpshot",
    POINT]
blocks = [r"^blocks", BLOCK]
defensive_rebound = [r"defensive rebound", REBOUND]
turnover = [r"\bturnover\b", TURNOVER]
shooting_foul = [r"shooting foul", FOUL]
makes_free_throw_one_of_one = [r"makes free throw 1 of 1", POINT]
misses_free_throw_one_of_one = [r"misses free throw 1 of 1", POINT]
makes_free_throw_one_of_two = [
    r"makes free throw 1 of 2"
    r"|makes free throw clear path 1 of 2", POINT]
makes_free_throw_two_of_two = [
    r"makes free throw 2 of 2"
    r"|makes free throw clear path 2 of 2", POINT]
makes_free_throw_one_of_three = [r"makes free throw 1 of 3", POINT]
makes_free_throw_two_of_three = [r"makes free throw 2 of 3", POINT]
makes_free_throw_three_of_three = [r"makes free throw 3 of 3", POINT]
misses_free_throw_one_of_two = [
    r"misses free throw 1 of 2"
    r"|misses free throw clear path 1 of 2", POINT]
misses_free_throw_two_of_two = [
    r"misses free throw 2 of 2"
    r"|misses free throw clear path 2 of 2", POINT]
misses_free_throw_one_of_three = [r"misses free throw 1 of 3", POINT]
misses_free_throw_two_of_three = [r"misses free throw 2 of 3", POINT]
misses_free_throw_three_of_three = [r"misses free throw 3 of 3", POINT]
offensive_rebound = [r"offensive rebound", REBOUND]
makes_driving_layup = [r"makes driving layup", POINT]
misses_driving_layup = [r"misses driving layup", POINT]
makes_layup = [r"makes layup"
               r"|makes (\d+)-foot layup", POINT]
misses_layup = [r"misses layup"
                r"|misses (\d+)-foot layup", POINT]
bad_pass = [r"bad pass", TURNOVER]
makes_driving_floating_jumpshot = [
    r"makes (\d+)-foot driving floating jump shot"
    r"|makes driving floating jump shot",
    POINT]
misses_driving_floating_jumpshot = [
    r"misses (\d+)-foot driving floating jump shot"
    r"|misses driving floating jump shot",
    POINT]
makes_three_point_pullup = [
    r"makes (\d+)-foot three point pullup jump shot"
    r"|makes three point pullup jump shot",
    POINT]
misses_three_point_pullup = [
    r"misses (\d+)-foot three point pullup jump shot"
    r"|misses three point pullup jump shot",
    POINT]
personal_foul = [r"personal foul", FOUL]
makes_driving_dunk = [r"makes (\d+)-foot driving dunk"
                      r"|makes driving dunk",
                      POINT]
misses_driving_dunk = [
    r"misses (\d+)-foot driving dunk"
    r"|misses driving dunk", POINT]
makes_alley_oop_dunk_shot = [
    r"makes (\d+)-foot alley oop dunk shot"
    r"|makes alley oop dunk shot",
    POINT]
misses_alley_oop_dunk_shot = [
    r"makes (\d+)-foot alley oop dunk shot"
    r"|misses alley oop dunk shot",
    POINT]
makes_running_pullup_jumpshot = [
    r"makes (\d+)-foot running pullup jump shot"
    r"|makes running pullup jump shot",
    POINT]
misses_running_pullup_jumpshot = [
    r"misses (\d+)-foot running pullup jump shot"
    r"|misses running pullup jump shot",
    POINT]
makes_stepback_jumpshot = [
    r"makes (\d+)-foot step back jumpshot"
    r"|makes step back jumpshot", POINT]
misses_stepback_jumpshot = [
    r"misses (\d+)-foot step back jumpshot"
    r"|misses step back jumpshot",
    POINT]
makes_tip_shot = [r"makes (\d+)-foot tip shot"
                  r"|makes tip shot", POINT]
misses_tip_shot = [r"misses (\d+)-foot tip shot"
                   r"|misses tip shot", POINT]
makes_alley_oop_layup = [r"makes alley oop layup", POINT]
misses_alley_oop_layup = [r"misses alley oop layup", POINT]
offensive_foul = [r"offensive foul", FOUL]
loose_ball_foul = [r"loose ball foul", FOUL]
makes_dunk = [r"makes (\d+)-foot dunk"
              r"|makes dunk"
              r"|makes slam dunk", POINT]
misses_dunk = [r"misses (\d+)-foot dunk"
               r"|misses dunk"
               r"|misses slam dunk",
               POINT]
traveling = [r"traveling"
             r"|raveling", TURNOVER]
makes_bank_shot = [r"makes (\d+)-foot jump bank shot"
                   r"|makes jump bank shot",
                   POINT]
makes_hook_shot = [r"makes (\d+)-foot hook shot"
                   r"|makes hook shot", POINT]
misses_hook_shot = [r"misses (\d+)-foot hook shot"
                    r"|misses hook shot", POINT]
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
technical_foul = [
    r"technical foul"
    r"|Players Technical"
    r"|defensive 3-seconds (technical foul)",
    FOUL]
misses_bank_shot = [
    r"misses (\d+)-foot jump bank shot"
    r"|misses jump bank shot", POINT]
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
makes_running_jumper = [
    r"makes (\d+)-foot running jumper"
    r"|makes running jumper", POINT]
misses_running_jumper = [
    r"misses (\d+)-foot running jumper"
    r"|misses running jumper", POINT]
defensive_team_rebound = [r"defensive team rebound", REBOUND]
team_rebound = [r"team rebound", REBOUND]
lost_ball = [r"lost ball", TURNOVER]
away_from_play_foul = [r"away from play foul", FOUL]
unspecified_foul = [r".*foul.*"
                    r"|Too Many Players Technical", FOUL]
makes_ft_flagrant_1_of_1 = [r"makes free throw flagrant 1 of 1", POINT]
misses_ft_flagrant_1_of_1 = [r"misses free throw flagrant 1 of 1", POINT]
unspecified_shot_clock_to = [r"shot clock turnover", TURNOVER]

play_types = [makes_three_point_jumper, misses_three_point_jumper,
                  makes_two_point, misses_two_point, makes_pullup,
                  misses_pullup, blocks, defensive_rebound, turnover,
                  shooting_foul, makes_free_throw_one_of_one,
                  misses_free_throw_one_of_one,
                  makes_free_throw_one_of_two, makes_free_throw_two_of_two,
                  makes_free_throw_one_of_three,
                  makes_free_throw_two_of_three,
                  makes_free_throw_three_of_three, misses_free_throw_one_of_two,
                  misses_free_throw_two_of_two,
                  misses_free_throw_one_of_three,
                  misses_free_throw_two_of_three,
                  misses_free_throw_three_of_three, offensive_rebound,
                  makes_driving_layup, misses_driving_layup,
                  makes_layup, misses_layup, bad_pass,
                  makes_driving_floating_jumpshot,
                  misses_driving_floating_jumpshot,
                  makes_three_point_pullup, misses_three_point_pullup,
                  personal_foul, makes_driving_dunk,
                  misses_driving_dunk,
                  makes_alley_oop_dunk_shot, misses_alley_oop_dunk_shot,
                  makes_running_pullup_jumpshot,
                  misses_running_pullup_jumpshot,
                  makes_stepback_jumpshot, misses_stepback_jumpshot,
                  makes_tip_shot, misses_tip_shot,
                  makes_alley_oop_layup,
                  misses_alley_oop_layup, offensive_foul, loose_ball_foul,
                  makes_dunk, misses_dunk, traveling,
                  makes_bank_shot,
                  makes_hook_shot, misses_hook_shot, kicked_ball_violation,
                  offensive_charge, violation,
                  makes_finger_roll_layup,
                  misses_finger_roll_layup, personal_take_foul,
                  transition_take_foul, defensive_three_seconds,
                  makes_technical_free_throw,
                  misses_technical_free_throw, hanging_techfoul, technical_foul,
                  misses_bank_shot, flagrant_foul_1,
                  makes_ft_flagrant_1_of_2,
                  makes_ft_flagrant_2_of_2, misses_ft_flagrant_1_of_2,
                  misses_ft_flagrant_2_of_2,
                  makes_ft_flagrant_1_of_3,
                  makes_ft_flagrant_2_of_3, makes_ft_flagrant_3_of_3,
                  misses_ft_flagrant_1_of_3,
                  misses_ft_flagrant_2_of_3, misses_ft_flagrant_3_of_3,
                  both_team_foul, ejected, makes_running_jumper,
                  misses_running_jumper, defensive_team_rebound,
                  team_rebound, lost_ball, away_from_play_foul,
                  unspecified_foul, makes_ft_flagrant_1_of_1,
                  misses_ft_flagrant_1_of_1, unspecified_shot_clock_to]