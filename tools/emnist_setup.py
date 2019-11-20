# dicts that return values to properly setup emnist
import string
def emnist_setup(SPLIT):
    # maps numerical labels to character labels
    labels = {'byclass': dict( 
            zip(range(62),list(range(10)) + list(string.ascii_uppercase) + list(string.ascii_lowercase))
                              ),
              'bymerge': dict(
                              zip(range(47),list(range(10)) + list(string.ascii_uppercase) +
                              ['a','b','d','e','f','g','h','n','q','r','t'])
                              ),
              'balanced': dict(
                               zip(range(47),list(range(10)) + list(string.ascii_uppercase) +
                               ['a','b','d','e','f','g','h','n','q','r','t'])
                               ),
              'letters': dict(zip(range(26),list(string.ascii_uppercase))),
              'digits': dict(zip(range(10),range(10))),
              'mnist': dict(zip(range(10),range(10)))
              }
    # number of classes
    classes = {'byclass':62, 'bymerge':47, 'balanced':47, 'letters':26, 'digits':10, 'mnist':10}
    # required transform on targets
    # needed for letters because they start from 1
    target_transform = {'byclass': None,
                        'bymerge': None,
                        'balanced': None,
                        'letters': lambda x:x-1,
                        'digits': None,
                        'mnist': None,
                        }
    return labels[SPLIT], classes[SPLIT], target_transform[SPLIT]