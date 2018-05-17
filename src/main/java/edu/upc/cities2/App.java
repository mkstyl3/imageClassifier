package edu.upc.cities2;

import java.io.File;

import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.imagefilter.PHOGFilter;

/**
 * Image Classifier
 *
 */

public class App 
{
    /**
     * Expects two parameters: training file and test file.
     *
     * @param args	the commandline arguments
     * @throws Exception	if something goes wrong
     */

    public static void main( String[] args )
    {
        final float TOTAL = 163;
        float oks = 0;
        try {

        //APLICAR EL FILTRO

        Instances trainingSubset = getFilteredDataSet("datasets/training.arff");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainingSubset);
        saver.setFile(new File("src/main/resources/fdatasets/filtered.arff"));
        saver.writeBatch();
        // train classifier
        NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
        //naive nbm = new naive();
        nbm.buildClassifier(trainingSubset);

        // output predictions
        System.out.println("# - actual - predicted - error - distribution");
        for (int i = 0; i < trainingSubset.numInstances(); i++) {
            double pred = nbm.classifyInstance(trainingSubset.instance(i));
            double[] dist = nbm.distributionForInstance(trainingSubset.instance(i));
            System.out.print((i+1));
            System.out.print(" - ");
            System.out.print(trainingSubset.instance(i).toString(trainingSubset.classIndex()));
            System.out.print(" - ");
            System.out.print(trainingSubset.classAttribute().value((int) pred));
            System.out.print(" - ");
            if (pred != trainingSubset.instance(i).classValue()) {
                System.out.print("yes");
            }
                
            else {
                System.out.print("no");
                oks++;
            }
                
            System.out.print(" - ");
            System.out.print(Utils.arrayToString(dist));
            System.out.println("");
        }
        System.out.println(String.valueOf(oks/TOTAL)+"%");
    } 
    
    catch (Exception ex) {
            System.out.println(ex.getMessage());
    }
}

    public static Instances getFilteredDataSet(String path) throws Exception {

        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        
        System.out.println(data.toSummaryString());
        
        PHOGFilter filter = new PHOGFilter();
        filter.setInputFormat(data);
        filter.setDebug(false);
        filter.setDoNotCheckCapabilities(false);
        
        filter.setImageDirectory("src/main/resources/images");
        Instances trainingSubset = Filter.useFilter(data, filter);

        String[] rem_options = new String[2];
        rem_options[0] = "-R";
        rem_options[1] = "1";
        Remove remove = new Remove();
        remove.setOptions(rem_options);
        remove.setInputFormat(trainingSubset);
        return Filter.useFilter(trainingSubset, remove);    
    }
}
