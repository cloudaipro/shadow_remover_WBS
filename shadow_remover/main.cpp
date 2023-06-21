//Note: This code is created by Bingshu Wang , 2019.08.31
//We proposed an effective method to remove shadows from document images.
//This is only for academic exchange. 
// If one wants use it for commercial purpose, please contact us right now by yb77408@umac.mo or philipchen@um.edu.mo.  
// or  https://www.fst.um.edu.mo/en/staff/pchen.html. 

// https://github.com/BookPlus2020/doc_shadow_removal/tree/master
//If you try to use this code, please cite our paper 
// "An Effective Background Estimation Method For Shadows Removal Of Document Images"  
// accepted by ICIP2019.
#include"doc_shadow_removal.h"
#include <filesystem>
namespace fs = std::filesystem;

void doShadowRemove(string fileName, string out_fileName)
{
    Mat img = imread(fileName);
    Mat result(img.size(), CV_8UC3, 3);
    ShadowRemoval(img, result);
    //imshow(fileName, img);
    //imshow(fileName + "_remove",result);
    imwrite(out_fileName, result);
}


int main(int argc, char** argv)
{
    std::string path = "./test";
    std::string outPath = "./Results/";

    for (const auto & entry : fs::directory_iterator(path)) {
        std::cout << entry.path() << std::endl;
        doShadowRemove(entry.path(), outPath + "" + (string)entry.path().filename() );
    }
    
    printf("done!");
    //getchar();
    //waitKey(0);
	return 0;
}







 

 
