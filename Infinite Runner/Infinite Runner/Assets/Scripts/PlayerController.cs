using UnityEngine;
using UnityEngine.UI;

public class PlayerController : MonoBehaviour
{
    public float moveSpeed = 8f;
    public float leftLimit = -4f;
    public float rightLimit = 4f;
    public float jumpForce = 7f;
    public Button startGameButton;
    public Button restartButton;

    private bool isGrounded = true;

    private Rigidbody rb;

    public bool gameOver = false;
    public bool gameStarted = false;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        if (!gameOver && gameStarted)
        {
            float horizontal =
                Input.GetAxis("Horizontal");

            Vector3 movement =
                Vector3.forward *
                horizontal *
                moveSpeed *
                Time.fixedDeltaTime;

            Vector3 newPosition =
                rb.position + movement;

            // Keep within road bounds
            newPosition.x =
                Mathf.Clamp(
                    newPosition.x,
                    leftLimit,
                    rightLimit
                );

            rb.MovePosition(newPosition);

            // Jump
            if (Input.GetKey(KeyCode.Space) && isGrounded)
            {
                rb.AddForce(
                    Vector3.up * jumpForce,
                    ForceMode.Impulse
                );

                isGrounded = false;
            }
        }
    }
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle"))
        {
            gameOver = true;
            gameStarted = false;
            restartButton.gameObject.SetActive(true);

        }

        if (collision.gameObject.CompareTag("Ground"))
        {
            isGrounded = true;
        }
    }

    public void StartGame()
    {
        gameStarted = true;
        gameOver = false;
        startGameButton.gameObject.SetActive(false);
        restartButton.gameObject.SetActive(false);
    }
}