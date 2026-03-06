package org.grobid.core.engines;

import org.junit.Test;
import org.junit.BeforeClass;
import org.grobid.core.main.LibraryLoader;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class AuthorParserReDoSTest {

    @BeforeClass
    public static void init() {
        LibraryLoader.load();
        try {
            // Warmup to load models
            AuthorParser parser = new AuthorParser();
            parser.processingCitation("warmup");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test(timeout = 20000) // Increase timeout for safety
    public void testReDoS() throws Exception {
        AuthorParser parser = new AuthorParser();
        StringBuilder sb = new StringBuilder();
        // Create a long string that might trigger backtracking if the regex was bad
        for (int i = 0; i < 50000; i++) {
            sb.append("et al. ");
        }
        String input = sb.toString();

        long start = System.currentTimeMillis();
        parser.processingCitation(input);
        long end = System.currentTimeMillis();

        System.out.println("Processing time: " + (end - start) + "ms");
        // Ensure it is reasonably fast
        assertTrue("Processing took too long: " + (end - start) + "ms", (end - start) < 2000);
    }

    @Test(timeout = 20000)
    public void testReDoS_NoMatch() throws Exception {
        AuthorParser parser = new AuthorParser();
        StringBuilder sb = new StringBuilder();
        // Create a long string with NO match
        for (int i = 0; i < 50000; i++) {
            sb.append("Smith Johnson ");
        }
        String input = sb.toString();

        long start = System.currentTimeMillis();
        parser.processingCitation(input);
        long end = System.currentTimeMillis();

        System.out.println("Processing time (no match): " + (end - start) + "ms");
        // Tokenizing a huge string might take time, but shouldn't be ReDoS-slow
        assertTrue("Processing took too long: " + (end - start) + "ms", (end - start) < 10000);
    }
}
