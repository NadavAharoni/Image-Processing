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

# Step 4
Create sample code for my students that calculates histogram, then calculates the cdf, then creates a new image using the cdf mapping.
Don't use cv2 functions, use pixelwise operations.
Place it in a file called "equalization_pixelwise.py" (or propose a better name if can think of such)


# Step 5
I want to demostrate how image normalization, if done in RGB space,
can ruin the colors.
We will use the function create_rects in "create_test_images.py".
We will create an image with width 400, height 300.
Background color will be (120, 20, 20), and two rectangles (not overlapping, they should each cover about 1/8 of the image), with RGB colors (140, 21, 21) and (150, 22, 22).
Now, if we will normalize each color separately, or even if we stretch the values around the mean by the same factor, the color, which is a shade of red, will become something different.
Then, I want to show the students that if we first convert to HSV or YCrCb, the color will be preserved.

So I would like a new python file, let's call it compare_normalize.py (you can suggest a better name).
In this file we will call create_rects in "create_test_images.py", and create an image as specified above.
Then please create code that "stretches" the values of the image in the following way:
1. Normalize such that min goes to 0, max to 255 - in each channel separately.
2. Multiply by a factor, let say of 5 around the mean - again each channel directly
3. Histogram equaliztion - again each channel directly.
4. Convert to HSV, then normalize just the brightness such that min goes to 0, max to 255, then convert back to RGB.
5. Convert to HSV, multiply each pixel's distance to the mean by 5. then convert back to RGB.
6. Convert to HSV, perform histogram equaliztion, then convert back to RGB.
7,8,9. Repeat steps 4,5,6 with YCrCb.
10. Display everything nicely with matplotlib. 


# suggestions

- Add a side-by-side plot of original + equalized images and their histograms.
- Add a short README and a small test saving expected files.