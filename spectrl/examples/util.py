import sys
import getopt


# Parse command line options
def parse_command_line_options():
    optval = getopt.getopt(sys.argv[1:], 'n:d:', [])
    itno = 0
    folder = ''
    for option in optval[0]:
        if option[0] == '-n':
            itno = int(option[1])
        if option[0] == '-d':
            folder = option[1]
    print('Run number: {}'.format(itno))
    print('Folder to save data: {}'.format(folder))
    return itno, folder
