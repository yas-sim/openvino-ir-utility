import sys
import argparse
import pickle

import numpy as np 

def check_err(a, b, err):
    if a>0:
        return b>(a*(100-err))/100 and b<(a*(100+err))/100
    elif a<0:
        return b<(a*(100-err))/100 and b>(a*(100+err))/100
    elif a==0:
        return a==b
    return None


def main(args):
    file_cmp = args.feature
    file_ref = args.reference_feature

    with open(file_cmp, 'rb') as f:
        features_cmp = pickle.load(f)

    with open(file_ref, 'rb') as f:
        features_ref = pickle.load(f)

    # Check all layers
    for layer_ref, feature_ref in features_ref.items():
        if layer_ref in features_cmp:
            feature_cmp = features_cmp[layer_ref]
            flat_ref = feature_ref[2].ravel()
            flat_cmp = feature_cmp[2].ravel()
            pass_ = 0
            error = 0
            nan   = 0
            inf   = 0
            print_cnt = 0
            # check individual values in the feature map
            for v_ref, v_cmp in zip(flat_ref, flat_cmp):
                if np.isnan(v_ref) or np.isnan(v_cmp):   # NaN ?
                    if args.verbose and print_cnt<args.top:
                        print(v_ref, ':', v_cmp)
                        print_cnt += 1
                    nan   += 1
                elif np.isinf(v_ref) or np.isinf(v_cmp): # inf ?
                    if args.verbose and print_cnt<args.top:
                        print(v_ref, ':', v_cmp)
                        print_cnt += 1
                    inf   += 1
                elif check_err(v_ref, v_cmp, args.error): # within error tolerance?
                    pass_ += 1
                else:
                    if args.verbose and print_cnt<args.top:
                        print(v_ref, ':', v_cmp)
                        print_cnt += 1
                    error += 1
            print('PASS:{:6}, ERROR:{:6}, E-Rate:{:6.2f}%, NaN:{:6}, Inf:{:6} - {}'.format(pass_, error, ((error/(pass_+error))*100), nan, inf, layer_ref))
        else:
            print('Layer \'{}\' doesn\'nt exist in {}'.format(layer_ref, file_cmp))

if __name__ == "__main__":
    print('*** OpenVINO feature map comparator')

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference_feature', type=str, required=True, help='input feature map pickle file (.pickle)')
    parser.add_argument('-f', '--feature', type=str, required=True, help='input feature map pickle file to compare with the reference feature map data (.pickle)')
    parser.add_argument('-e', '--error', type=float, default=10, help='error tolerance (%). default=10')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='display error values')
    parser.add_argument('-t', '--top', type=int, default=5, help='# of error values to display (per layer, with -v option)')
    args = parser.parse_args()

    print('Error tolerance : {}%'.format(args.error))
    sys.exit(main(args))
