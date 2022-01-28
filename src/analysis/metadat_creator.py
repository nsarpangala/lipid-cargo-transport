import sys
import datetime
def initialize_metadata(path):
    import socket
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    f = open(path+'.txt', "w")
    f.write('Git previous commit %s \n'%sha)
    f.write('Host computer: %s \n'%(str(socket.gethostname())))
    today = datetime.date.today()
    f.write('Date and Time: %s \n'%(today.ctime()))
    f.write('Name of the file: %s \n'%(str(sys.argv[0])))
    return f