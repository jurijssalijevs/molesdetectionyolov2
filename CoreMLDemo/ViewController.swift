
import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    private let imageView: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(systemName: "photo")
        imageView.contentMode = .scaleAspectFit
        return imageView
    }()

    private let label: UILabel = {
        let label = UILabel()
        label.textAlignment = .center
        label.text = "Select Image"
        label.numberOfLines = 0
        return label
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.addSubview(label)
        view.addSubview(imageView)

        let tap = UITapGestureRecognizer(
            target: self,
            action: #selector(didTapImage)
        )
        tap.numberOfTapsRequired = 1
        imageView.isUserInteractionEnabled = true
        imageView.addGestureRecognizer(tap)
    }

    @objc func didTapImage() {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        present(picker, animated: true)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        imageView.frame = CGRect(
            x: 20,
            y: view.safeAreaInsets.top,
            width: view.frame.size.width - 40,
            height: view.frame.size.width - 40
        )
        label.frame = CGRect(
            x: 20,
            y: view.safeAreaInsets.top + (view.frame.size.width - 40) + 10,
            width: view.frame.size.width - 40,
            height: 100
        )
    }

    private func analyzeImage(image: UIImage?) {
        guard let image = image else {
            return
        }

        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else {
            return
        }

        do {
            let config = MLModelConfiguration()
            let coreMLModel = try Skincancer(configuration: config)
            let model = try VNCoreMLModel(for: coreMLModel.model)

            // Create a Vision request
            let request = VNCoreMLRequest(model: model) { [weak self] (request, error) in
                guard let self = self else { return }
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        // Draw bounding boxes on the image
                        let annotatedImage = self.drawBoundingBoxes(on: image, with: results)
                        self.imageView.image = annotatedImage
                        self.label.text = "Detected \(results.count) mole(s)"
                    }
                } else {
                    DispatchQueue.main.async {
                        self.label.text = "No moles detected"
                    }
                }
            }

            // Configure the request
            request.imageCropAndScaleOption = .scaleFill

            // Create a handler and perform the request
            let handler = VNImageRequestHandler(cgImage: cgImage, orientation: image.imageOrientation.cgImagePropertyOrientation, options: [:])
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                do {
                    try handler.perform([request])
                } catch {
                    print("Failed to perform image request: \(error.localizedDescription)")
                    DispatchQueue.main.async {
                        self?.label.text = "Error: \(error.localizedDescription)"
                    }
                }
            }
        } catch {
            print("Failed to load ML model: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.label.text = "Error loading model: \(error.localizedDescription)"
            }
            return
        }
    }

    private func drawBoundingBoxes(on image: UIImage, with observations: [VNRecognizedObjectObservation]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(at: CGPoint.zero)

        let context = UIGraphicsGetCurrentContext()
        context?.setStrokeColor(UIColor.red.cgColor)
        context?.setLineWidth(2.0)

        let imageSize = image.size

        for observation in observations {
            let boundingBox = observation.boundingBox

            // Convert bounding box to image coordinates
            let rect = CGRect(
                x: boundingBox.origin.x * imageSize.width,
                y: (1 - boundingBox.origin.y - boundingBox.size.height) * imageSize.height,
                width: boundingBox.size.width * imageSize.width,
                height: boundingBox.size.height * imageSize.height
            )

            context?.stroke(rect)

            // Optionally, draw the label and confidence
            if let label = observation.labels.first {
                let text = "\(label.identifier) (\(String(format: "%.2f", label.confidence * 100))%)"
                let textAttributes: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: 12),
                    .foregroundColor: UIColor.red
                ]
                let textRect = CGRect(x: rect.origin.x, y: rect.origin.y - 15, width: rect.width, height: 15)
                text.draw(in: textRect, withAttributes: textAttributes)
            }
        }

        let annotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return annotatedImage ?? image
    }

    // MARK: - UIImagePickerControllerDelegate

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        // Cancelled
        picker.dismiss(animated: true, completion: nil)
    }

    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
        picker.dismiss(animated: true, completion: nil)
        guard let image = info[.originalImage] as? UIImage else {
            return
        }
        imageView.image = image
        analyzeImage(image: image)
    }
}

// MARK: - Extensions

extension UIImage.Orientation {
    var cgImagePropertyOrientation: CGImagePropertyOrientation {
        switch self {
            case .up: return .up
            case .upMirrored: return .upMirrored
            case .down: return .down
            case .downMirrored: return .downMirrored
            case .left: return .left
            case .leftMirrored: return .leftMirrored
            case .right: return .right
            case .rightMirrored: return .rightMirrored
            @unknown default:
                fatalError("Unknown orientation")
        }
    }
}
