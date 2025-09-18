// export-annotations.groovy
//
// Script to export annotations from each image in a
// QuPath project and write them to the file
// <image>.annotations.json.
//
// This script is not called directly, but passed in
// using the export-annotations.sh shell script.
//
// Author: Lawrence Holder, Washington State University

def imageName = getProjectEntry().getImageName()
def fileName = imageName + '.annotations.json'
def annotations = getAnnotationObjects()
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
def output = gson.toJson(annotations)
new File(fileName).text = output
