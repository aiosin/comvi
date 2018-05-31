from dimred import dimred
import pip


def main():
    _dependencies = ['numpy','matplotlib','tensorflow','scipy','sklearn','scikit-image','mahotas','opencv-python','requests']
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    for package in _dependencies:
        pipmain(['install','--user', package])
    dimred.main()


if __name__ == '__main__':
    main()
