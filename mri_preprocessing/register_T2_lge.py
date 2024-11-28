import SimpleITK as sitk

# Load the fixed and moving images
fixed_image = sitk.ReadImage("path/to/fixed_image.dcm")
moving_image = sitk.ReadImage("path/to/moving_image.dcm")

# Initialize the registration method
registration_method = sitk.ImageRegistrationMethod()

# Set the similarity metric (Mutual Information is common for DICOM)
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

# Set optimizer (Gradient Descent)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

# Use an affine transform as the initial transform
initial_transform = sitk.AffineTransform(fixed_image.GetDimension())
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Set the interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

# Execute the registration
final_transform = registration_method.Execute(fixed_image, moving_image)

# Apply the transformation to the moving image
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(final_transform)
registered_image = resampler.Execute(moving_image)

# Save the registered image
sitk.WriteImage(registered_image, "path/to/registered_image.dcm")

print("Final metric value: ", registration_method.GetMetricValue())
print("Optimizer's stopping condition: ", registration_method.GetOptimizerStopConditionDescription())
