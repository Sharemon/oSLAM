#include <iostream>
#include "DataLoader.hpp"

using namespace oSLAM;
using namespace cv;
using namespace std;


int main(int, char**) {

    DataLoader data_loader("/home/panxiaoyu/Desktop/oSLAM/data/unit_test");
    Mat rgb, depth;
    bool ret;
    ret = data_loader.pop(rgb, depth);
    ret = data_loader.pop(rgb, depth);

    imshow("./rgb.png", rgb);
    imshow("./depth.png", depth);

    waitKey(0);

    cout << "Hello, oSLAM!" << endl;

}
