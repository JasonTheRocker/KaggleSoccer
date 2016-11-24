import sqlite3
import numpy as np
from operator import itemgetter

def getMatches():
	conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
	f = open('./matches.txt', 'w+')
	res = c.execute("""SELECT * FROM match_data""").fetchall()
	for row in res:
        	print >>f,  row
	return res

def getPlayers():
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        f = open('./players.txt', 'w+')
	res = c.execute("""SELECT * FROM player_data""").fetchall()
        for row in res:
                print >>f, row
	return res

def getTeams():
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        f = open('./teams.txt', 'w+')
	res = c.execute("""SELECT * FROM team_data""").fetchall()
        for row in res:
                print >>f, row
	return res

def getTeamAtt(table, api_id, date):
	for r in table:
		if (api_id == r[1] and date >= r[2]):
			return r
	return None

def getPlayerAtt(table, api_id, date):
	for r in table:
		if (api_id == r[1] and date >= r[2]):
			return r
	return []


def getFeatures():
	#genMatchData()
        #genPlayerData()
        #genTeamData()

	matches = getMatches()
	players = getPlayers()
	teams = getTeams()
	
	
	
	sorted_teams = sorted(teams, key = itemgetter(2), reverse=True)
	sorted_players = sorted(players, key = itemgetter(2), reverse=True)
	
	marray = matches
	tarray = np.array(matches)
	features = []
	f = open('./st.txt', 'w+')
	
	for i in marray:
		players = ()
		all_check = True
		for p in range(56, 78):
			ret = getPlayerAtt(sorted_players, int(i[56]), i[5])
			ret = ret[3:]
			if len(ret) > 0:
				players = players + ret
			else:
				all_check = False
				break
		if all_check:
			toadd = i + players
			print >>f, toadd
			features.append(toadd)
	
	
	

def genMatchData():
	conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
	c.execute("""DROP TABLE IF EXISTS match_data""")
	
	attlist = {'country_id','league_id','season','stage','m_date','match_api_id','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal','home_player_X1','home_player_X2','home_player_X3','home_player_X4','home_player_X5','home_player_X6','home_player_X7','home_player_X8','home_player_X9','home_player_X10','home_player_X11','away_player_X1','away_player_X2','away_player_X3','away_player_X4','away_player_X5','away_player_X6','away_player_X7','away_player_X8','away_player_X9','away_player_X10','away_player_X11','home_player_Y1','home_player_Y2','home_player_Y3','home_player_Y4','home_player_Y5','home_player_Y6','home_player_Y7','home_player_Y8','home_player_Y9','home_player_Y10','home_player_Y11','away_player_Y1','away_player_Y2','away_player_Y3','away_player_Y4','away_player_Y5','away_player_Y6','away_player_Y7','away_player_Y8','away_player_Y9','away_player_Y10','away_player_Y11','home_player_1','home_player_2','home_player_3','home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10','home_player_11','away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6','away_player_7','away_player_8','away_player_9','away_player_10','away_player_11'}	
	
	nullCheck = "(id IS NOT NULL)"

        for s in attlist:
                nullCheck = nullCheck + "AND (" + s + " IS NOT NULL)"

	c.execute("""
	CREATE TABLE match_data AS
        SELECT id, country_id, league_id, season, stage, date AS m_date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_X1, home_player_X2, home_player_X3, home_player_X4, home_player_X5, home_player_X6, home_player_X7, home_player_X8, home_player_X9, home_player_X10, home_player_X11, away_player_X1, away_player_X2, away_player_X3, away_player_X4, away_player_X5, away_player_X6, away_player_X7, away_player_X8, away_player_X9, away_player_X10, away_player_X11, home_player_Y1, home_player_Y2, home_player_Y3, home_player_Y4, home_player_Y5, home_player_Y6, home_player_Y7, home_player_Y8, home_player_Y9, home_player_Y10, home_player_Y11, away_player_Y1, away_player_Y2, away_player_Y3, away_player_Y4, away_player_Y5, away_player_Y6, away_player_Y7, away_player_Y8, away_player_Y9, away_player_Y10, away_player_Y11, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 
        FROM Match 
        WHERE """ + nullCheck)

def genPlayerData():
	conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
	c.execute("""DROP TABLE IF EXISTS player_data""")

        attlist = {'player_api_id','p_date','overall_rating','potential','preferred_foot','attacking_work_rate','defensive_work_rate','crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve','free_kick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility','reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression','interceptions','positioning','vision','penalties','marking','standing_tackle','sliding_tackle','gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes','height','weight'}
        nullCheck = "(player_fifa_api_id IS NOT NULL)"

        for s in attlist:
                nullCheck = nullCheck + "AND (" + s + " IS NOT NULL)"

        c.execute("""
	CREATE TABLE player_data AS
        SELECT *
        FROM
        (SELECT player_fifa_api_id, player_api_id, date AS p_date, overall_rating, potential, preferred_foot, attacking_work_rate, defensive_work_rate, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, marking, standing_tackle, sliding_tackle, gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes
        FROM Player_attributes)
        NATURAL JOIN
        (SELECT player_api_id, player_fifa_api_id, height, weight
        FROM Player)
        WHERE """ + nullCheck)

def genTeamData():
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        c.execute("""DROP TABLE IF EXISTS team_data""")

	attlist = {'id','team_fifa_api_id','team_api_id','date','buildUpPlaySpeed','buildUpPlaySpeedClass','buildUpPlayDribbling','buildUpPlayDribblingClass','buildUpPlayPassing','buildUpPlayPassingClass','buildUpPlayPositioningClass','chanceCreationPassing','chanceCreationPassingClass','chanceCreationCrossing','chanceCreationCrossingClass','chanceCreationShooting','chanceCreationShootingClass','chanceCreationPositioningClass','defencePressure','defencePressureClass','defenceAggression','defenceAggressionClass','defenceTeamWidth','defenceTeamWidthClass','defenceDefenderLineClass'}
	nullCheck = "(id IS NOT NULL)"
        for s in attlist:
                nullCheck = nullCheck + "AND (" + s + " IS NOT NULL)"

	c.execute("""
        CREATE TABLE team_data AS
	SELECT team_fifa_api_id, team_api_id, date AS t_date, buildUpPlaySpeed, buildUpPlaySpeedClass, buildUpPlayDribbling, buildUpPlayDribblingClass, buildUpPlayPassing, buildUpPlayPassingClass, buildUpPlayPositioningClass, chanceCreationPassing, chanceCreationPassingClass, chanceCreationCrossing, chanceCreationCrossingClass, chanceCreationShooting, chanceCreationShootingClass, chanceCreationPositioningClass, defencePressure, defencePressureClass, defenceAggression, defenceAggressionClass, defenceTeamWidth, defenceTeamWidthClass, defenceDefenderLineClass
	FROM Team_Attributes
	WHERE """ + nullCheck)

if __name__ == "__main__":
	getFeatures()
