import capture
import os
import sys

def main():
	for _ in range(100):
		try:
			capture.run(sys.argv[1:])
		except TypeError:
			break
		# python training.py -r agents/blinky-clyde/myTeam.py -b agents/team_template/myTeam.py --time 10000

if __name__ == '__main__':
	main()