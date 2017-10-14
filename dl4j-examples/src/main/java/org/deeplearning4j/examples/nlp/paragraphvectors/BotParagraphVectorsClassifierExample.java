package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * This is basic example for documents classification done with DL4j ParagraphVectors.
 * The overall idea is to use ParagraphVectors in the same way we use LDA:
 * topic space modelling.
 * <p/>
 * In this example we assume we have few labeled categories that we can use
 * for training, and few unlabeled documents. And our goal is to determine,
 * which category these unlabeled documents fall into
 * <p/>
 * <p/>
 * Please note: This example could be improved by using learning cascade
 * for higher accuracy, but that's beyond basic example paradigm.
 *
 * @author raver119@gmail.com
 */
public class BotParagraphVectorsClassifierExample {

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(BotParagraphVectorsClassifierExample.class);

    public static void main(String[] args) throws Exception {
        BotParagraphVectorsClassifierExample app = new BotParagraphVectorsClassifierExample();
        app.makeParagraphVectors();
        app.checkUnlabeledData();
    }

    void makeParagraphVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("anuhakvec/labeled");

        // build a iterator for our dataset
        iterator = new FileLabelAwareIterator.Builder()
            .addSourceFolder(resource.getFile())
            .build();

        tokenizerFactory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 10);
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
            .learningRate(0.025)
            .minLearningRate(0.001)
            .batchSize(1000)
            .epochs(1000)
            //.iterations(10000)
            .iterate(iterator)
            .trainWordVectors(true)
            .tokenizerFactory(tokenizerFactory)
            .minWordFrequency(1)
            .useAdaGrad(false)
            .layerSize(1000)
            .allowParallelTokenization(true)
            .useUnknown(false)
            .build();

        // Start model training
        paragraphVectors.fit();

    }

    void checkUnlabeledData() throws FileNotFoundException {
      /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */
        ClassPathResource unClassifiedResource = new ClassPathResource("anuhakvec/labeled");
        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
            .addSourceFolder(unClassifiedResource.getFile())
            .build();

        MeansBuilder meansBuilder = new MeansBuilder(
            (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
            tokenizerFactory);

        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
            (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        int total = 0;
        int correct = 0;
        while (unClassifiedIterator.hasNextDocument()) {
            try{
                LabelledDocument document = unClassifiedIterator.nextDocument();
                INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
                List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

                Collections.sort(scores, new Comparator<Pair<String, Double>>() {
                    @Override
                    public int compare(final Pair<String, Double> lhs, Pair<String, Double> rhs) {
                        //TODO return 1 if rhs should be before lhs
                        if (lhs.getSecond() > rhs.getSecond())
                            return -1;
                        else if (lhs.getSecond() < rhs.getSecond())
                            return 1;
                        else
                            return 0;
                    }
                });

                log.info("Document '" + document.getLabels() + " -  " + document.getContent() + "' falls into the following categories: ");
                int index = 0;
                for (Pair<String, Double> score : scores) {
                    log.info("        " + score.getFirst() + ": " + score.getSecond());
                    index++;
                    if (index > 5) {
                        break;
                    }
                }

                total++;

                if(scores.get(0).getFirst().equals(document.getLabel())){
                    correct++;
                }
            }catch (Exception e){
                System.out.println(e);
            }

        }

        System.out.println(correct+"/"+total+"="+(correct/(double)total));

    }
}
