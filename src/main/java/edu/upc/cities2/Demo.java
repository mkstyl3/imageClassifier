package edu.upc.cities2;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.imagefilter.PHOGFilter;

public class Demo {
    /**
     * Expects two parameters: training file and test file.
     *
     * @param args	the commandline arguments
     * @throws Exception	if something goes wrong
     */

    public static void main( String[] args ) {
        try {
            final float TOTAL = 163;
            float oks = 0;
            Classifier cls = (Classifier) SerializationHelper.read("src/main/resources/models/nbm.model");

            //APLICAR EL FILTRO
    
            Instances demoSubset = getFilteredDataSet("datasets/demo.arff");
            ArffSaver saver = new ArffSaver();
            saver.setInstances(demoSubset);
            saver.setFile(new File("src/main/resources/fdatasets/demoFiltered.arff"));
            saver.writeBatch();
           
           // output predictions
            System.out.println("# - actual - predicted - error - distribution");
            
                for (int i = 0; i < demoSubset.numInstances(); i++) {
                    double pred = cls.classifyInstance(demoSubset.instance(i));
                    double[] dist = cls.distributionForInstance(demoSubset.instance(i));
                    System.out.print((i+1));
                    System.out.print(" - ");
                    System.out.print(demoSubset.instance(i).toString(demoSubset.classIndex()));
                    System.out.print(" - ");
                    System.out.print(demoSubset.classAttribute().value((int) pred));
                    System.out.print(" - ");
                    if (pred != demoSubset.instance(i).classValue()) {
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
            
            
            
            catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    private static String summary(Evaluation eval){
        return Utils.doubleToString(eval.correct(), 12, 4) + "\t " +
                Utils.doubleToString(eval.pctCorrect(), 12, 4) + "%";
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
        
        filter.setImageDirectory("src/main/resources/images/demo");
        Instances demoSubset = Filter.useFilter(data, filter);

        String[] rem_options = new String[2];
        rem_options[0] = "-R";
        rem_options[1] = "1";
        Remove remove = new Remove();
        remove.setOptions(rem_options);
        remove.setInputFormat(demoSubset);
        return Filter.useFilter(demoSubset, remove);    
    }
}


