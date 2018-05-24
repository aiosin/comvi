from dimred import dimred
import pip

_dependencies = ['numpy','matplotlib','tensorflow','scipy','sklearn','skimage','mahotas','cv2','requests']

def main():
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    for package in _dependencies:
        pipmain(['install','--user', package])
    dimred.main()


if __name__ == '__main__':
    main()
