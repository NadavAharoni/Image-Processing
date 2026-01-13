# Step 1

I want to create images that will demonstrate histogram equalization to students.
I want a "multi modal" histogram, with two or more "peaks".
My idea is to create a function: draw_gradient_rect, which will recieve an image to draw on, rectange coordinates, and two  colors, let's call them c1, c2.
The function will draw a filled rectangle in which the top left corner will have the color c1, and the bottom right corner will have the color c2. The other pixels will have an interpolated value.
The function should work for color images and for greyscale. It should check that the shape of the image matches the colors. For a greyscale image the colors will be one value (0..255), and for color images each color will be a tuple (of length 3).
Then we will create image, for example with the following rectangles:
1. A rectangle (A) covering the whole image with colors c1=50, c2=60.
2. A rectangle (B) covering par of the image with c1=70, c2=80
3. A rectangle (C) with c1=180, c2=185
4. A rectangle (D) with c1=195, c2=200
D should be a sub-area of C. B & C should not intersect.

I believe that the histogram for such image will have one peak for eah rectangle.
Regular normalization will strech the histogram, but histogram equalization will do much better.

Please let me know what you think about this plan.
If it sounds good, please create the function draw_gradient_rect in the file "create_test_images.py" (unless there is already an opencv function that does this, in such case please point me to it).
Then, let's create a function that creates the example image.
I suggest that it will recieve a list of dictionaries. Each dictionary will have an entry for a "rect" and an entry for "colors".
The "rect" entry can be a tuple containing two tuples, one for top-left and one for bottom-right.
The colors entry will be a tuple of two colors, which will either be greyscale levels or a tuple for RGB.

# Step 2:
Now let's create a color image.
Here, I want to demonstrate that histogram equzliation that be done for each channel seperately, can change the colors in an undesired way.
I think that this can be done by having different ranges for each color.
For example, we can have c1=(50,20,30), c2=(55,40,35). In the example the histogram for the middle color (green) is more stretched, so histgram equalization will strech the other colors more. I think that this will result in a modification of the colors.
But, if we first convert the color image to HSV, or YCbCr, perform histgram equalization just on the brightness and then go back to RGB, the results will look better.

If you agree with this analysis please add code to create an example color image.

# Step 3

I already have a file in this directory "cdf.py" which reads an image, shows a histogram and the cdf.
Please change it to perform equalization and display the image, the histogram, the CDF, the histogram and cdf after equalization, and then the resulting image.
If the image is a color image, let's display each color in a separate row.
In each row, display one channel (but in color, not greyscale), the histogram, CFD, equalized histogram and cdf and the resulting channel.
Add a row also for the color image.
If it's greyscale, create just one "row" in the plot.

# suggestions

- Add a side-by-side plot of original + equalized images and their histograms.
- Add a short README and a small test saving expected files.